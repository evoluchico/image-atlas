"""Serve stage: open a built dataset bundle, never recompute anything.

Endpoints (DESIGN.md section 6):
  GET  /api/manifest
  POST /api/filter      {"where": "<sql where clause>"} -> {token, count}
  POST /api/select      {"polygon": [[x,y],...], "base_token": "..."} -> {token, count}
  GET  /api/viewport?z=&x0=&y0=&x1=&y1=&token=
  GET  /api/image/<id>
  GET  /api/export?token=             (CSV of metadata + x,y for the token's set)
  GET  /atlases/..., /previews/...   (immutable bundle assets)
  GET  /                              (frontend)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import FORMAT_VERSION, summarize

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_TILES_PER_QUERY = 4096
ALL_TOKEN = "all"
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".webp": "image/webp",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


class DatasetError(SystemExit):
    pass


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized ray casting. pts (N,2), poly (M,2) -> bool (N,)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        crosses = (y0 > y) != (y1 > y)
        if crosses.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < xint)
        x0, y0 = x1, y1
    return inside


class Dataset:
    """Read-only view over a dataset bundle. Validates on open, fails loudly."""

    def __init__(self, root: Path):
        self.root = root.resolve()

        def need(rel: str) -> Path:
            p = self.root / rel
            if not p.exists():
                raise DatasetError(
                    f"dataset {self.root} is missing '{rel}'. "
                    f"Re-run `atlas build`; the GUI never rebuilds bundles."
                )
            return p

        self.manifest = json.loads(need("manifest.json").read_text())
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise DatasetError(
                f"unsupported format_version {self.manifest.get('format_version')!r} "
                f"(this build of atlas supports {FORMAT_VERSION})"
            )
        self.n = int(self.manifest["count"])
        self.threshold = int(self.manifest["aggregate_threshold"])
        self.zmin = int(self.manifest["zoom"]["min"])
        self.zmax = int(self.manifest["zoom"]["max"])

        self.xy = np.load(need("points/xy.npy"), mmap_mode="r")
        self.rep = np.load(need("points/rep.npy"), mmap_mode="r")
        self.tiles = {}
        for z in range(self.zmin, self.zmax + 1):
            self.tiles[z] = (
                np.load(need(f"points/z{z}_order.npy"), mmap_mode="r"),
                np.load(need(f"points/z{z}_keys.npy")),
                np.load(need(f"points/z{z}_starts.npy")),
                np.load(need(f"points/z{z}_rep.npy")),
            )
        for name, arr, want in (
            ("points/xy.npy", self.xy, (self.n, 2)),
            ("points/rep.npy", self.rep, (self.n,)),
        ):
            if tuple(arr.shape) != want:
                raise DatasetError(f"{name} has shape {arr.shape}, expected {want}")
        n_pages = len(list((self.root / "atlases").glob("atlas_*.webp")))
        if n_pages != self.manifest["atlas"]["pages"]:
            raise DatasetError(
                f"found {n_pages} atlas pages, manifest says {self.manifest['atlas']['pages']}"
            )

        self.db_path = need("metadata.sqlite")
        with self._connect() as con:
            (rows,) = con.execute("SELECT COUNT(*) FROM images").fetchone()
        if rows != self.n:
            raise DatasetError(f"metadata.sqlite has {rows} rows, manifest says {self.n}")

        self._filters: OrderedDict[str, tuple] = OrderedDict()  # token -> (mask, count, where)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        con.execute("PRAGMA query_only = ON")
        return con

    # ----------------------------------------------------------- filters

    def make_filter(self, where: str) -> tuple[str, int]:
        where = (where or "").strip().rstrip(";")
        if not where:
            return ALL_TOKEN, self.n
        token = hashlib.sha1(where.encode()).hexdigest()[:12]
        with self._lock:
            hit = self._filters.get(token)
            if hit:
                self._filters.move_to_end(token)
                return token, hit[1]
        con = self._connect()
        try:
            ids = np.fromiter(
                (r[0] for r in con.execute(f"SELECT id FROM images WHERE ({where})")),
                dtype=np.int64,
            )
        finally:
            con.close()
        ids = ids[(ids >= 0) & (ids < self.n)]
        mask = np.zeros(self.n, dtype=bool)
        mask[ids] = True
        with self._lock:
            self._filters[token] = (mask, len(ids), where)
            while len(self._filters) > 32:
                self._filters.popitem(last=False)
        return token, len(ids)

    def make_selection(self, polygon, base_token: str) -> tuple[str, int]:
        poly = np.asarray(polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
            raise ValueError("polygon must be a list of at least 3 [x, y] points")
        base = self.get_mask(base_token)  # may raise KeyError
        token = "sel" + hashlib.sha1(
            poly.tobytes() + base_token.encode()
        ).hexdigest()[:9]
        with self._lock:
            hit = self._filters.get(token)
            if hit:
                self._filters.move_to_end(token)
                return token, hit[1]
        xy = np.asarray(self.xy, dtype=np.float64)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        cand = np.flatnonzero(
            (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0])
            & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
            & (base if base is not None else True)
        )
        mask = np.zeros(self.n, dtype=bool)
        if len(cand):
            mask[cand[points_in_polygon(xy[cand], poly)]] = True
        count = int(mask.sum())
        with self._lock:
            self._filters[token] = (mask, count, "<lasso>")
            while len(self._filters) > 32:
                self._filters.popitem(last=False)
        return token, count

    def export_csv(self, token: str) -> bytes:
        mask = self.get_mask(token)  # may raise KeyError
        buf = io.StringIO()
        writer = csv.writer(buf)
        con = self._connect()
        try:
            cur = con.execute("SELECT * FROM images ORDER BY id")
            writer.writerow([d[0] for d in cur.description] + ["x", "y"])
            xy = self.xy
            for row in cur:
                img_id = row[0]
                if mask is None or mask[img_id]:
                    writer.writerow(
                        list(row) + [f"{xy[img_id, 0]:.6f}", f"{xy[img_id, 1]:.6f}"]
                    )
        finally:
            con.close()
        return buf.getvalue().encode()

    def get_mask(self, token: str):
        if token in ("", ALL_TOKEN):
            return None
        with self._lock:
            hit = self._filters.get(token)
            if hit:
                self._filters.move_to_end(token)
                return hit[0]
        raise KeyError(token)

    # ---------------------------------------------------------- viewport

    def viewport(self, z: int, x0: float, y0: float, x1: float, y1: float, token: str) -> dict:
        mask = self.get_mask(token)
        z = max(self.zmin, min(self.zmax, z))

        def span(zz):
            side = 1 << zz
            ax0 = max(0, min(side - 1, int(np.floor(x0 * side))))
            ax1 = max(0, min(side - 1, int(np.floor(x1 * side))))
            ay0 = max(0, min(side - 1, int(np.floor(y0 * side))))
            ay1 = max(0, min(side - 1, int(np.floor(y1 * side))))
            return ax0, ax1, ay0, ay1

        tx0, tx1, ty0, ty1 = span(z)
        while z > self.zmin and (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > MAX_TILES_PER_QUERY:
            z -= 1
            tx0, tx1, ty0, ty1 = span(z)

        order, keys, starts, tile_reps = self.tiles[z]
        side = 1 << z
        tile_size = 1.0 / side
        xy, rep = self.xy, self.rep
        aggregates, items = [], []

        def emit_items(ids):
            pos = xy[ids]
            items.extend(
                {"id": int(s), "x": float(px), "y": float(py)}
                for s, (px, py) in zip(ids.tolist(), pos.tolist())
            )

        for ty in range(ty0, ty1 + 1):
            base = ty * side
            i0 = int(np.searchsorted(keys, base + tx0, side="left"))
            i1 = int(np.searchsorted(keys, base + tx1, side="right"))
            for i in range(i0, i1):
                key = int(keys[i])
                s0, s1 = int(starts[i]), int(starts[i + 1])
                if mask is None:
                    # unfiltered fast path: representative was precomputed
                    cnt = s1 - s0
                    if cnt <= self.threshold:
                        emit_items(np.asarray(order[s0:s1]))
                        continue
                    rid = int(tile_reps[i])
                    aggregates.append(
                        {"tx": key % side, "ty": key // side, "count": cnt,
                         "id": rid, "x": float(xy[rid, 0]), "y": float(xy[rid, 1])}
                    )
                    continue
                members = order[s0:s1]
                surv = members[mask[members]]
                cnt = len(surv)
                if cnt == 0:
                    continue
                if cnt <= self.threshold:
                    emit_items(surv)
                    continue
                samp = summarize.sample(surv)
                pos = np.asarray(xy[samp], dtype=np.float64)
                w = np.asarray(rep[samp], dtype=np.float64)
                best = summarize.pick_representative(pos, w, tile_size)
                aggregates.append(
                    {"tx": key % side, "ty": key // side, "count": cnt,
                     "id": int(samp[best]),
                     "x": float(pos[best][0]), "y": float(pos[best][1])}
                )
        return {"z": z, "aggregates": aggregates, "items": items}

    # ------------------------------------------------------------- image

    def image_info(self, img_id: int) -> dict | None:
        if not (0 <= img_id < self.n):
            return None
        con = self._connect()
        try:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM images WHERE id = ?", (img_id,)).fetchone()
        finally:
            con.close()
        info = dict(row) if row else {"id": img_id}
        info["preview_url"] = f"/previews/{img_id // 1000:03d}/{img_id:08d}.webp"
        return info


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    ds: Dataset  # set on the subclass created in serve()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    # helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message):
        self._json({"error": message}, code)

    def _file(self, path: Path, immutable=False):
        if not path.is_file():
            return self._error(404, "not found")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        if immutable:
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _safe_join(root: Path, rel: str) -> Path | None:
        p = (root / rel.lstrip("/")).resolve()
        return p if p.is_relative_to(root) else None

    # routes
    def do_GET(self):
        url = urlparse(self.path)
        route = url.path
        try:
            if route == "/" or route == "/index.html":
                return self._file(FRONTEND_DIR / "index.html")
            if route in ("/app.js", "/style.css"):
                return self._file(FRONTEND_DIR / route.lstrip("/"))
            if route == "/api/manifest":
                return self._json(self.ds.manifest)
            if route == "/api/viewport":
                q = parse_qs(url.query)

                def f(name, cast=float):
                    return cast(q[name][0])

                try:
                    result = self.ds.viewport(
                        f("z", int), f("x0"), f("y0"), f("x1"), f("y1"),
                        q.get("token", [ALL_TOKEN])[0],
                    )
                except KeyError:
                    return self._error(410, "unknown filter token; re-apply the filter")
                except (ValueError, IndexError):
                    return self._error(400, "bad viewport parameters")
                return self._json(result)
            if route == "/api/export":
                token = parse_qs(url.query).get("token", [ALL_TOKEN])[0]
                try:
                    body = self.ds.export_csv(token)
                except KeyError:
                    return self._error(410, "unknown token; re-apply the filter/selection")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="atlas-export.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route.startswith("/api/image/"):
                try:
                    img_id = int(route.rsplit("/", 1)[1])
                except ValueError:
                    return self._error(400, "bad image id")
                info = self.ds.image_info(img_id)
                return self._json(info) if info else self._error(404, "no such image")
            if route.startswith("/atlases/") or route.startswith("/previews/"):
                p = self._safe_join(self.ds.root, route)
                return self._file(p, immutable=True) if p else self._error(403, "forbidden")
            return self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/api/filter", "/api/select"):
            return self._error(404, "not found")
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._error(400, "body must be JSON")
        if route == "/api/filter":
            try:
                token, count = self.ds.make_filter(payload.get("where", ""))
            except sqlite3.Error as e:
                return self._error(400, f"SQL error: {e}")
        else:
            try:
                token, count = self.ds.make_selection(
                    payload.get("polygon"), payload.get("base_token", ALL_TOKEN)
                )
            except KeyError:
                return self._error(410, "unknown base token; re-apply the filter")
            except (ValueError, TypeError) as e:
                return self._error(400, str(e))
        return self._json({"token": token, "count": count})


def run(args) -> None:
    ds = Dataset(Path(args.dataset))
    handler = type("BoundHandler", (Handler,), {"ds": ds})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving '{ds.manifest['name']}' ({ds.n} images) at {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


def add_parser(sub) -> None:
    p = sub.add_parser("serve", help="serve a built dataset bundle locally")
    p.add_argument("dataset", help="dataset bundle directory (output of `atlas build`)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.set_defaults(func=run)
