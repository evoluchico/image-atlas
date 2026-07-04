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
import gc
import hashlib
import io
import json
import os
import sqlite3
import sys
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from . import FORMAT_VERSION, summarize

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
MAX_TILES_PER_QUERY = 4096
MAX_SPRITES_PER_STRIP = 1024
STRIP_COLS = 32  # must match frontend/app.js
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
        self._maps = []   # open memory-maps, tracked so close() can release them
        try:
            self._open()
        except Exception:
            self.close()   # release any partial handles so the dir can be deleted
            raise

    def _track(self, arr):
        self._maps.append(arr)
        return arr

    def close(self) -> None:
        """Release all memory-maps. Required on Windows before the bundle
        directory can be moved or deleted (an open map locks the file).

        A numpy memmap array exports a pointer to its mmap, so the mmap can't
        be closed until every referencing array is dropped — hence we clear the
        array references and gc first, then close the (now unexported) maps."""
        mmaps = [getattr(a, "_mmap", None) for a in self._maps]
        self._maps = []
        self.xy = self.rep = self.sprites = None
        self.tiles = {}
        gc.collect()
        for mm in mmaps:
            if mm is not None:
                try:
                    mm.close()
                except (ValueError, OSError, BufferError):
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _open(self) -> None:
        def need(rel: str) -> Path:
            p = self.root / rel
            if not p.exists():
                raise DatasetError(
                    f"dataset {self.root} is missing '{rel}'. "
                    f"Re-run `atlas build`; the GUI never rebuilds bundles."
                )
            return p

        self.manifest = json.loads(need("manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise DatasetError(
                f"unsupported format_version {self.manifest.get('format_version')!r} "
                f"(this build of atlas supports {FORMAT_VERSION})"
            )
        self.n = int(self.manifest["count"])
        self.threshold = int(self.manifest["aggregate_threshold"])
        self.zmin = int(self.manifest["zoom"]["min"])
        self.zmax = int(self.manifest["zoom"]["max"])

        self.xy = self._track(np.load(need("points/xy.npy"), mmap_mode="r"))
        self.rep = self._track(np.load(need("points/rep.npy"), mmap_mode="r"))
        self.tiles = {}
        for z in range(self.zmin, self.zmax + 1):
            self.tiles[z] = (
                self._track(np.load(need(f"points/z{z}_order.npy"), mmap_mode="r")),
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

        self.cell = int(self.manifest["sprite_cell"])
        sprites_path = need("sprites.bin")
        want_size = self.n * self.cell * self.cell * 3
        if sprites_path.stat().st_size != want_size:
            raise DatasetError(
                f"sprites.bin is {sprites_path.stat().st_size} bytes, expected {want_size}"
            )
        self.sprites = self._track(np.memmap(
            sprites_path, dtype=np.uint8, mode="r", shape=(self.n, self.cell, self.cell, 3)
        ))
        self._strips: OrderedDict[str, bytes] = OrderedDict()

        self.db_path = need("metadata.sqlite")
        with self._connect() as con:
            (rows,) = con.execute("SELECT COUNT(*) FROM images").fetchone()
        if rows != self.n:
            raise DatasetError(f"metadata.sqlite has {rows} rows, manifest says {self.n}")

        self._filters: OrderedDict[str, tuple] = OrderedDict()  # token -> (mask, count, where)
        self._grid_cache: dict[str, list] = {}   # search token -> top ranked ids
        self._axes: OrderedDict[str, dict] = OrderedDict()  # axis token -> record
        self._lock = threading.Lock()

        # optional search: CLIP text->image and/or OCR full-text
        self.search_cfg = self.manifest.get("search")
        self._searcher = None
        with self._connect() as con:
            self.has_ocr = bool(con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ocr'"
            ).fetchone())
        self.has_search = bool(self.search_cfg) or self.has_ocr
        # semantic axes need the image embeddings (not just OCR)
        self.has_axis = bool(self.search_cfg)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        con.execute("PRAGMA query_only = ON")
        return con

    # ----------------------------------------------------------- filters

    @property
    def user_columns(self) -> list[dict]:
        return self.manifest.get("metadata_columns", [])

    def columns_summary(self) -> list[dict]:
        """Per user column: numeric range, small value list, or free-text —
        enough for the UI to render sliders/dropdowns instead of asking for SQL."""
        hidden = set(self.manifest.get("filter_hide", []))
        con = self._connect()
        try:
            out = []
            for c in self.user_columns:
                name, typ = c["name"], c["type"]
                if name in hidden:   # no auto control; still usable via Advanced SQL
                    continue
                if typ in ("INTEGER", "REAL"):
                    lo, hi = con.execute(
                        f'SELECT MIN("{name}"), MAX("{name}") FROM images'
                    ).fetchone()
                    if lo is not None:
                        out.append({"name": name, "kind": "range", "min": lo, "max": hi})
                    continue
                (distinct,) = con.execute(
                    f'SELECT COUNT(DISTINCT "{name}") FROM images'
                ).fetchone()
                if distinct <= 60:
                    vals = [r[0] for r in con.execute(
                        f'SELECT DISTINCT "{name}" FROM images '
                        f'WHERE "{name}" IS NOT NULL ORDER BY "{name}"'
                    )]
                    out.append({"name": name, "kind": "choice", "values": vals})
                else:
                    out.append({"name": name, "kind": "text"})
            return out
        finally:
            con.close()

    def _store_ids(self, token: str, ids: np.ndarray, where: str) -> tuple[str, int]:
        ids = ids[(ids >= 0) & (ids < self.n)]
        mask = np.zeros(self.n, dtype=bool)
        mask[ids] = True
        with self._lock:
            self._filters[token] = (mask, len(ids), where)
            while len(self._filters) > 32:
                self._filters.popitem(last=False)
        return token, int(len(ids))

    def _run_ids(self, sql: str, params=()) -> np.ndarray:
        con = self._connect()
        try:
            return np.fromiter(
                (r[0] for r in con.execute(sql, params)), dtype=np.int64
            )
        finally:
            con.close()

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
        return self._store_ids(token, self._run_ids(
            f"SELECT id FROM images WHERE ({where})"), where)

    def make_filter_structured(self, filters: list) -> tuple[str, int]:
        """Build a parameterized WHERE from UI controls — no user SQL.
        Each filter: {col, values:[...]} for choices, {col, min, max} for ranges."""
        allowed = {c["name"] for c in self.user_columns}
        clauses, params = [], []
        for f in filters or []:
            col = f.get("col")
            if col not in allowed:
                continue
            if f.get("values"):
                vals = list(f["values"])
                clauses.append(f'"{col}" IN ({",".join("?" * len(vals))})')
                params += vals
            if f.get("min") is not None:
                clauses.append(f'"{col}" >= ?')
                params.append(f["min"])
            if f.get("max") is not None:
                clauses.append(f'"{col}" <= ?')
                params.append(f["max"])
        if not clauses:
            return ALL_TOKEN, self.n
        where = " AND ".join(clauses)
        token = "s" + hashlib.sha1((where + repr(params)).encode()).hexdigest()[:11]
        with self._lock:
            hit = self._filters.get(token)
            if hit:
                self._filters.move_to_end(token)
                return token, hit[1]
        return self._store_ids(
            token, self._run_ids(f"SELECT id FROM images WHERE {where}", params), where)

    # --------------------------------------------------------- search

    def _ocr_ids(self, query: str, limit: int) -> list[int]:
        # quote each token so FTS5 treats them literally (avoids syntax errors)
        terms = " ".join(f'"{t}"' for t in query.split() if t)
        if not terms:
            return []
        con = self._connect()
        try:
            return [r[0] for r in con.execute(
                "SELECT rowid FROM ocr WHERE ocr MATCH ? ORDER BY rank LIMIT ?",
                (terms, limit),
            )]
        except sqlite3.Error:
            return []
        finally:
            con.close()

    GRID_IDS = 120   # ranked ids returned for the side-panel results grid

    def search(self, query: str, base_token: str = ALL_TOKEN,
               mode: str = "fused", k: int = 1000) -> tuple[str, int, list, bool]:
        """Text search restricted to base_token's set (composes with filters).
        mode='text': OCR full-text only — an *exact* set with a true count.
        mode='fused': CLIP + OCR fused by RRF — the top k by relevance.
        Returns (token, count, top_ranked_ids, exact)."""
        query = (query or "").strip()
        if not query:
            base_count = (self.n if base_token in ("", ALL_TOKEN)
                          else int(self.get_mask(base_token).sum()))
            return base_token, base_count, [], True
        base = self.get_mask(base_token)   # None = all; may raise KeyError
        keep = (lambda i: True) if base is None else (lambda i: bool(base[i]))
        token = "q" + hashlib.sha1(
            (mode + "|" + query + "|" + base_token).encode()).hexdigest()[:11]
        with self._lock:
            hit = self._filters.get(token)
            if hit:
                return (token, hit[1], list(self._grid_cache.get(token, [])),
                        mode == "text")

        if mode == "text":   # exact: OCR matches only, ranked by relevance
            ordered = [i for i in self._ocr_ids(query, 20000) if keep(i)]
            tok, count = self._store_ids(token, np.array(ordered, np.int64), f"text:{query}")
            grid = ordered[: self.GRID_IDS]
            with self._lock:
                self._grid_cache[token] = grid
            return tok, count, grid, True

        # fused: Reciprocal Rank Fusion of CLIP and OCR rankings
        pool = max(k * 4, 2000)
        scores: dict[int, float] = {}
        if self.search_cfg:
            ranked, _ = self._ensure_searcher().rank(query)
            rank = 0
            for i in ranked.tolist():
                if keep(i):
                    scores[i] = scores.get(i, 0.0) + 1.0 / (60 + rank)
                    rank += 1
                    if rank >= pool:
                        break
        if self.has_ocr:
            rank = 0
            for i in self._ocr_ids(query, pool * 3):
                if keep(i):
                    scores[i] = scores.get(i, 0.0) + 2.0 / (60 + rank)  # weight literal text
                    rank += 1
        ordered = sorted(scores, key=scores.get, reverse=True)[:k]
        tok, count = self._store_ids(token, np.array(ordered, np.int64), f"search:{query}")
        grid = ordered[: self.GRID_IDS]
        with self._lock:
            self._grid_cache[token] = grid
        return tok, count, grid, False

    # --------------------------------------------------------- semantic axes

    def _ensure_searcher(self):
        if self._searcher is None:
            if not self.search_cfg:
                raise DatasetError("this dataset has no image embeddings")
            from .search import TextSearcher
            self._searcher = TextSearcher(self.root, self.search_cfg["model"])
        return self._searcher

    def _end_ids(self, spec: dict) -> np.ndarray:
        """Union of a text-free end's sources: hand-picked ids + token masks."""
        ids: set[int] = {int(i) for i in spec.get("ids", []) if 0 <= int(i) < self.n}
        for t in spec.get("tokens", []):
            m = self.get_mask(t)             # may raise KeyError (unknown token)
            if m is None:
                ids.update(range(self.n))    # 'all' token
            else:
                ids.update(int(i) for i in np.flatnonzero(m))
        return np.fromiter(ids, dtype=np.int64)

    def _end_vector(self, spec: dict) -> np.ndarray:
        """Resolve one axis end to a unit vector in embedding space."""
        if spec.get("text"):
            v = self._ensure_searcher().encode_texts([spec["text"]])[0]
        else:
            E = self._ensure_searcher().ensure_matrix()
            idx = self._end_ids(spec)
            if not len(idx):
                raise ValueError("axis end is empty")
            rows = E[idx]
            rows = rows[np.linalg.norm(rows, axis=1) > 0]   # drop missing embeddings
            if not len(rows):
                raise ValueError("axis end has no embedded images")
            v = rows.mean(axis=0)
        n = np.linalg.norm(v)
        return v / (n if n else 1.0)

    @staticmethod
    def _has_end(spec) -> bool:
        return bool(spec) and bool(spec.get("text") or spec.get("ids") or spec.get("tokens"))

    def make_axis(self, x_spec: dict, y_spec: dict | None, base_token: str) -> tuple[str, dict]:
        """Build a 1- or 2-axis projection over the embeddings. Cached by token.
        overlay (x only): per-image normalized score for map tint + spectrum.
        scatter (x and y): an ephemeral (sX, sY) layout tiled like the main map."""
        base = self.get_mask(base_token)     # None = all; may raise KeyError
        E = self._ensure_searcher().ensure_matrix()
        baseidx = slice(None) if base is None else np.flatnonzero(base)

        def one_axis(pair: dict) -> dict:
            a = self._end_vector(pair["a"])
            if self._has_end(pair.get("b")):
                b = self._end_vector(pair["b"])
                d, mid, two = a - b, (a + b) / 2.0, True
            else:
                d, mid, two = a.copy(), None, False
            nd = np.linalg.norm(d)
            d = d / (nd if nd else 1.0)
            s = (E @ d).astype(np.float32)
            p2, p98 = (float(v) for v in np.percentile(s[baseidx], [2, 98]))
            rng = (p98 - p2) or 1.0
            norm = np.clip((s - p2) / rng, 0.0, 1.0).astype(np.float32)
            div = float(np.clip((float(mid @ d) - p2) / rng, 0.0, 1.0)) if two else None
            return {"norm": norm, "p2": p2, "p98": p98, "div": div}

        xr = one_axis(x_spec)
        yr = one_axis(y_spec) if self._has_end((y_spec or {}).get("a")) else None
        record = {"mode": "scatter" if yr else "overlay", "x": xr, "y": yr}
        if yr:
            from . import build
            # invert Y so the high-score end (A/C) is at the TOP of the screen
            axy = np.column_stack([xr["norm"], 1.0 - yr["norm"]]).astype(np.float32)
            record["layout"] = {
                "xy": axy,
                "tiles": build.compute_tile_indexes(axy, np.asarray(self.rep), self.zmax),
            }
        token = "ax" + hashlib.sha1(
            repr([x_spec, y_spec, base_token]).encode()).hexdigest()[:11]
        with self._lock:
            self._axes[token] = record
            while len(self._axes) > 4:      # each scatter layout is ~tens of MB
                self._axes.popitem(last=False)
        return token, record

    def get_axis(self, token: str) -> dict | None:
        if not token:
            return None
        with self._lock:
            rec = self._axes.get(token)
            if rec:
                self._axes.move_to_end(token)
            return rec

    def axis_spectrum(self, record: dict, base_token: str, k: int = 24) -> list:
        """Overlay only: one best-thumbnail representative per score bin, A->B."""
        norm = record["x"]["norm"]
        base = self.get_mask(base_token)
        idx = np.arange(self.n) if base is None else np.flatnonzero(base)
        if not len(idx):
            return []
        rep = np.asarray(self.rep)
        edges = np.linspace(0.0, 1.0, k + 1)
        out = []
        for bi in range(k):
            lo, hi = edges[bi], edges[bi + 1]
            inb = (norm[idx] >= lo) & ((norm[idx] < hi) if bi < k - 1 else (norm[idx] <= hi))
            cand = idx[inb]
            if not len(cand):
                continue
            best = int(cand[np.argmax(rep[cand])])
            out.append({"id": best, "pos": float((lo + hi) / 2)})
        return out

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

    def export_csv(self, token: str, axis_token: str = ALL_TOKEN) -> bytes:
        mask = self.get_mask(token)  # may raise KeyError
        axis = self.get_axis(axis_token)
        extra = ["x", "y"]
        if axis is not None:
            extra.append("axis_x")
            if axis["mode"] == "scatter":
                extra += ["axis_y", "quadrant"]
        xnorm = axis["x"]["norm"] if axis is not None else None
        ynorm = axis["y"]["norm"] if (axis is not None and axis["mode"] == "scatter") else None
        divx = (axis["x"]["div"] if axis is not None else None)
        divx = 0.5 if divx is None else divx
        divy = (axis["y"]["div"] if ynorm is not None else None)
        divy = 0.5 if divy is None else divy
        buf = io.StringIO()
        writer = csv.writer(buf)
        con = self._connect()
        try:
            cur = con.execute("SELECT * FROM images ORDER BY id")
            writer.writerow([d[0] for d in cur.description] + extra)
            xy = self.xy
            for row in cur:
                img_id = row[0]
                if not (mask is None or mask[img_id]):
                    continue
                cells = list(row) + [f"{xy[img_id, 0]:.6f}", f"{xy[img_id, 1]:.6f}"]
                if xnorm is not None:
                    cells.append(f"{float(xnorm[img_id]):.6f}")
                if ynorm is not None:
                    nx, ny = float(xnorm[img_id]), float(ynorm[img_id])
                    quad = ("x+" if nx >= divx else "x-") + " " + ("y+" if ny >= divy else "y-")
                    cells += [f"{ny:.6f}", quad]
                writer.writerow(cells)
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

    def viewport(self, z: int, x0: float, y0: float, x1: float, y1: float,
                 token: str, axis_token: str = ALL_TOKEN) -> dict:
        mask = self.get_mask(token)
        z = max(self.zmin, min(self.zmax, z))

        # A scatter axis relayout swaps in its own coords + tiles; an overlay
        # axis keeps the map layout but tints markers by a per-image score.
        axis = self.get_axis(axis_token)
        scatter = axis is not None and axis["mode"] == "scatter"
        tiles_by_z = axis["layout"]["tiles"] if scatter else self.tiles
        xy = axis["layout"]["xy"] if scatter else self.xy
        rep = self.rep
        score = axis["x"]["norm"] if (axis is not None and not scatter) else None

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

        order, keys, starts, tile_reps = tiles_by_z[z]
        side = 1 << z
        tile_size = 1.0 / side
        aggregates, items = [], []

        def emit_items(ids):
            pos = xy[ids]
            for s, (px, py) in zip(ids.tolist(), pos.tolist()):
                it = {"id": int(s), "x": float(px), "y": float(py)}
                if score is not None:
                    it["score"] = float(score[s])
                items.append(it)

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
                    ag = {"tx": key % side, "ty": key // side, "count": cnt,
                          "id": rid, "x": float(xy[rid, 0]), "y": float(xy[rid, 1])}
                    if score is not None:
                        ag["score"] = float(score[order[s0:s1]].mean())
                    aggregates.append(ag)
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
                ag = {"tx": key % side, "ty": key // side, "count": cnt,
                      "id": int(samp[best]),
                      "x": float(pos[best][0]), "y": float(pos[best][1])}
                if score is not None:
                    ag["score"] = float(score[surv].mean())
                aggregates.append(ag)
        return {"z": z, "aggregates": aggregates, "items": items}

    def tile_members(self, z: int, tx: int, ty: int, token: str,
                     offset: int, limit: int, axis_token: str = ALL_TOKEN) -> dict:
        """Filtered members of one tile, best representatives first."""
        mask = self.get_mask(token)
        z = max(self.zmin, min(self.zmax, z))
        axis = self.get_axis(axis_token)
        tiles_by_z = (axis["layout"]["tiles"]
                      if axis is not None and axis["mode"] == "scatter" else self.tiles)
        order, keys, starts, _ = tiles_by_z[z]
        side = 1 << z
        key = ty * side + tx
        i = int(np.searchsorted(keys, key))
        if i >= len(keys) or keys[i] != key:
            return {"total": 0, "ids": []}
        members = order[starts[i] : starts[i + 1]]
        surv = members[mask[members]] if mask is not None else np.asarray(members)
        ranked = surv[np.argsort(-np.asarray(self.rep[surv]), kind="stable")]
        page = ranked[offset : offset + limit]
        return {"total": int(len(surv)), "ids": [int(s) for s in page]}

    # ------------------------------------------------------------ sprites

    def sprite_strip(self, ids: list[int]) -> bytes:
        """Pack the requested sprites into one WebP, STRIP_COLS per row.

        This is what makes remote viewing cheap: a viewport transfers only
        the sprites it shows (~1-2 KB each), never a shared atlas page.
        """
        key = ",".join(map(str, ids))
        with self._lock:
            hit = self._strips.get(key)
            if hit:
                self._strips.move_to_end(key)
                return hit
        c = self.cell
        cols = min(STRIP_COLS, len(ids))
        rows = (len(ids) + cols - 1) // cols
        canvas = np.zeros((rows * c, cols * c, 3), np.uint8)
        for i, img_id in enumerate(ids):
            r, col = divmod(i, cols)
            canvas[r * c : (r + 1) * c, col * c : (col + 1) * c] = self.sprites[img_id]
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, "WEBP", quality=80)
        body = buf.getvalue()
        with self._lock:
            self._strips[key] = body
            while len(self._strips) > 32:
                self._strips.popitem(last=False)
        return body

    def warm_sprites(self) -> None:
        """Sequentially touch sprites.bin to pull it into the OS page cache
        (random 7 KB reads on a cold spinning disk would dominate strip
        latency otherwise). Runs in a daemon thread at startup."""
        try:
            with open(self.root / "sprites.bin", "rb") as f:
                while f.read(1 << 24):
                    pass
        except OSError:
            pass

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
        else:
            # frontend files: never cache, so a server upgrade can't leave a
            # stale app.js talking to a newer API
            self.send_header("Cache-Control", "no-cache, no-store")
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
                return self._json({**self.ds.manifest, "has_search": self.ds.has_search,
                                   "has_axis": self.ds.has_axis})
            if route == "/api/search":
                q = parse_qs(url.query)
                mode = "text" if q.get("mode", ["fused"])[0] == "text" else "fused"
                try:
                    token, count, ids, exact = self.ds.search(
                        q.get("q", [""])[0], q.get("base", [ALL_TOKEN])[0], mode)
                except KeyError:
                    return self._error(410, "unknown base token; re-apply the filter")
                except Exception as e:
                    return self._error(500, f"search failed: {e}")
                return self._json({"token": token, "count": count, "ids": ids, "exact": exact})
            if route == "/api/labels":
                p = self.ds.root / "labels.json"
                if p.exists():
                    return self._file(p)
                return self._json({"labels": []})
            if route == "/api/columns":
                return self._json({"columns": self.ds.columns_summary()})
            if route == "/density.webp":
                p = self.ds.root / "density.webp"
                return self._file(p, immutable=True) if p.exists() else self._error(404, "no density")
            if route == "/api/contours":
                p = self.ds.root / "density_contours.json"
                return self._file(p) if p.exists() else self._json({"levels": []})
            if route == "/api/viewport":
                q = parse_qs(url.query)

                def f(name, cast=float):
                    return cast(q[name][0])

                try:
                    result = self.ds.viewport(
                        f("z", int), f("x0"), f("y0"), f("x1"), f("y1"),
                        q.get("token", [ALL_TOKEN])[0],
                        q.get("axis", [ALL_TOKEN])[0],
                    )
                except KeyError:
                    return self._error(410, "unknown filter token; re-apply the filter")
                except (ValueError, IndexError):
                    return self._error(400, "bad viewport parameters")
                return self._json(result)
            if route == "/api/tile":
                q = parse_qs(url.query)
                try:
                    result = self.ds.tile_members(
                        int(q["z"][0]), int(q["tx"][0]), int(q["ty"][0]),
                        q.get("token", [ALL_TOKEN])[0],
                        max(0, int(q.get("offset", ["0"])[0])),
                        min(500, max(1, int(q.get("limit", ["60"])[0]))),
                        q.get("axis", [ALL_TOKEN])[0],
                    )
                except KeyError as e:
                    if str(e).strip("'") in ("z", "tx", "ty"):
                        return self._error(400, "need z, tx, ty")
                    return self._error(410, "unknown filter token; re-apply the filter")
                except (ValueError, IndexError):
                    return self._error(400, "bad tile parameters")
                return self._json(result)
            if route == "/api/sprites":
                raw = parse_qs(url.query).get("ids", [""])[0]
                try:
                    ids = [int(s) for s in raw.split(",") if s]
                except ValueError:
                    return self._error(400, "ids must be comma-separated integers")
                if not ids or len(ids) > MAX_SPRITES_PER_STRIP:
                    return self._error(400, f"need 1..{MAX_SPRITES_PER_STRIP} ids")
                if any(i < 0 or i >= self.ds.n for i in ids):
                    return self._error(400, "id out of range")
                body = self.ds.sprite_strip(ids)
                self.send_response(200)
                self.send_header("Content-Type", "image/webp")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(body)
                return
            if route == "/api/export":
                q = parse_qs(url.query)
                token = q.get("token", [ALL_TOKEN])[0]
                try:
                    body = self.ds.export_csv(token, q.get("axis", [ALL_TOKEN])[0])
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
            if route.startswith("/previews/"):
                p = self._safe_join(self.ds.root, route)
                return self._file(p, immutable=True) if p else self._error(403, "forbidden")
            return self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _do_axis(self, payload):
        if not self.ds.has_axis:
            return self._error(400, "this dataset has no image embeddings for axes")
        x = payload.get("x") or {}
        y = payload.get("y")
        base = payload.get("base_token", ALL_TOKEN)
        if not Dataset._has_end((x or {}).get("a")):
            return self._error(400, "axis X needs at least end A")
        try:
            token, rec = self.ds.make_axis(x, y, base)
        except KeyError:
            return self._error(410, "unknown selection token; re-apply it")
        except (ValueError, DatasetError) as e:
            return self._error(400, str(e))
        except Exception as e:
            return self._error(500, f"axis failed: {e}")

        def label(spec, default):
            return (spec.get("label") or spec.get("text") or default) if spec else None

        labels = {"x": {"a": label(x.get("a"), "A"), "b": label(x.get("b"), "B")}}
        resp = {"token": token, "mode": rec["mode"], "labels": labels,
                "divX": rec["x"]["div"], "divY": None}
        if rec["mode"] == "scatter":
            labels["y"] = {"a": label((y or {}).get("a"), "C"),
                           "b": label((y or {}).get("b"), "D")}
            # world Y is inverted (top = high score), so flip the divider too
            resp["divY"] = None if rec["y"]["div"] is None else 1.0 - rec["y"]["div"]
        else:
            resp["stats"] = {"p2": rec["x"]["p2"], "p98": rec["x"]["p98"]}
            resp["spectrum"] = self.ds.axis_spectrum(rec, base)
        return self._json(resp)

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/api/filter", "/api/select", "/api/axis"):
            return self._error(404, "not found")
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._error(400, "body must be JSON")
        if route == "/api/axis":
            return self._do_axis(payload)
        if route == "/api/filter":
            try:
                if "filters" in payload:
                    token, count = self.ds.make_filter_structured(payload["filters"])
                else:
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
    server = ThreadingHTTPServer((args.host, args.port), handler)
    threading.Thread(target=ds.warm_sprites, daemon=True).start()
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving '{ds.manifest['name']}' ({ds.n} images) at {url}  (Ctrl-C to stop)")
    if args.host not in ("127.0.0.1", "localhost"):
        print(
            "WARNING: bound to a non-loopback interface. The server has no "
            "authentication; anyone who can reach this host can browse the dataset."
        )
    # Windows and macOS desktops always have a display; only Linux/X11 needs a
    # display variable, whose absence means a headless server (e.g. over SSH).
    headless = sys.platform not in ("win32", "darwin") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if headless and not args.no_browser:
        print(
            f"(headless session — not opening a browser. From your own machine:\n"
            f"   ssh -L {args.port}:localhost:{args.port} <user>@<this-host>\n"
            f" then open http://localhost:{args.port}/ locally)"
        )
    elif not args.no_browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


def add_parser(sub) -> None:
    p = sub.add_parser("serve", help="serve a built dataset bundle locally")
    p.add_argument("dataset", help="dataset bundle directory (output of `atlas build`)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default 127.0.0.1; use 0.0.0.0 for LAN "
                        "access — unauthenticated, prefer ssh -L port forwarding)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.set_defaults(func=run)
