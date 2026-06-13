"""Build stage: raw images (+ optional metadata/coords) -> dataset bundle.

The bundle layout and every constant here are specified in DESIGN.md
(section 4 and 5). The build is explicit and offline; `atlas serve`
never recomputes any of this.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import re
import shutil
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import FORMAT_VERSION, labels, summarize

Image.MAX_IMAGE_PIXELS = None  # local, user-owned data

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
FEAT_SIZE = 8  # 8x8 RGB features for fallback layout + quality score
DENSITY_BINS = 512


# ---------------------------------------------------------------- scanning

def find_images(root: Path) -> list[str]:
    rels = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    rels.sort()  # deterministic id assignment
    return rels


# ------------------------------------------------------------- per image

def _process_one(task):
    """Decode once; emit sprite pixels, detail preview file, feature vector.

    Runs in worker processes. Corrupt files become gray placeholders with
    quality 0 so that ids stay dense (see DESIGN.md section 5.2).
    """
    img_id, abspath, preview_path, cell, preview_max = task
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(abspath) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = preview_max / max(w, h)
            pv = (
                im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
                if scale < 1.0
                else im
            )
            pv.save(preview_path, "WEBP", quality=80)
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            sq = im.crop((left, top, left + s, top + s))
            sprite = np.asarray(sq.resize((cell, cell), Image.LANCZOS), dtype=np.uint8)
            feat = np.asarray(sq.resize((FEAT_SIZE, FEAT_SIZE), Image.LANCZOS), dtype=np.float32) / 255.0
            quality = 1.0 if float(feat.var()) > 1e-4 else 0.25
            return img_id, sprite.tobytes(), feat.tobytes(), w, h, quality
    except Exception:
        try:
            Image.fromarray(np.full((64, 64, 3), 96, np.uint8)).save(preview_path, "WEBP")
        except Exception:
            pass
        sprite = np.full((cell, cell, 3), 96, np.uint8)
        feat = np.full((FEAT_SIZE, FEAT_SIZE, 3), 0.5, np.float32)
        return img_id, sprite.tobytes(), feat.tobytes(), 0, 0, 0.0


def preview_relpath(img_id: int) -> str:
    return f"previews/{img_id // 1000:03d}/{img_id:08d}.webp"


# ------------------------------------------------------------ coordinates

def load_coords(coords_csv: Path, rels: list[str]) -> np.ndarray:
    with open(coords_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"coords file {coords_csv} is empty")
    cols = list(rows[0].keys())
    xcol = next((c for c in cols if c.lower() == "x"), cols[1])
    ycol = next((c for c in cols if c.lower() == "y"), cols[2])
    keycol = next((c for c in cols if c not in (xcol, ycol)), cols[0])
    table = {}
    for r in rows:
        try:
            val = (float(r[xcol]), float(r[ycol]))
        except (TypeError, ValueError):
            continue
        k = (r[keycol] or "").strip()
        for alias in (k, Path(k).name, Path(k).stem):
            table.setdefault(alias, val)
    xy = np.empty((len(rels), 2), np.float64)
    missing = []
    for i, rel in enumerate(rels):
        hit = table.get(rel) or table.get(Path(rel).name) or table.get(Path(rel).stem)
        if hit is None:
            missing.append(rel)
        else:
            xy[i] = hit
    if missing:
        raise SystemExit(
            f"coords file covers {len(rels) - len(missing)}/{len(rels)} images; "
            f"missing e.g. {missing[:3]}. Coordinates must cover every image."
        )
    return xy


def _rank01(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), np.float64)
    r[order] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def pca_layout(feats: np.ndarray) -> np.ndarray:
    """Fallback 2D layout: PCA of tiny color features, rank-spread per axis."""
    X = feats - feats.mean(axis=0)
    cov = X.T @ X
    _, vecs = np.linalg.eigh(cov)
    proj = X @ vecs[:, -2:][:, ::-1]  # top two components
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 1e-9, proj.shape)
    return np.stack([_rank01(proj[:, a] + jitter[:, a]) for a in range(2)], axis=1)


def normalize_coords(xy: np.ndarray) -> np.ndarray:
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (0.005 + 0.99 * (xy - lo) / span).astype(np.float32)


# ----------------------------------------------------------------- scores

def density_scores(xy: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=DENSITY_BINS, range=[[0, 1], [0, 1]])
    for _ in range(2):  # light box blur
        p = np.pad(hist, 1, mode="edge")
        hist = sum(
            p[1 + dy : DENSITY_BINS + 1 + dy, 1 + dx : DENSITY_BINS + 1 + dx]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ) / 9.0
    ix = np.minimum((xy[:, 0] * DENSITY_BINS).astype(int), DENSITY_BINS - 1)
    iy = np.minimum((xy[:, 1] * DENSITY_BINS).astype(int), DENSITY_BINS - 1)
    return _rank01(hist[ix, iy]).astype(np.float32)


# ------------------------------------------------------------ tile index

def write_tile_indexes(points_dir: Path, xy: np.ndarray, rep: np.ndarray, max_zoom: int) -> None:
    n = len(xy)
    xy64 = xy.astype(np.float64)
    rep64 = rep.astype(np.float64)
    for z in range(max_zoom + 1):
        side = 1 << z
        tx = np.minimum((xy[:, 0] * side).astype(np.int64), side - 1)
        ty = np.minimum((xy[:, 1] * side).astype(np.int64), side - 1)
        key = (ty * side + tx).astype(np.uint32)
        order = np.argsort(key, kind="stable").astype(np.uint32)
        sorted_keys = key[order]
        keys, idx = np.unique(sorted_keys, return_index=True)
        starts = np.append(idx, n).astype(np.uint32)
        # unfiltered per-tile representative, precomputed (stable under pan/zoom)
        tile_size = 1.0 / side
        rep_ids = np.empty(len(keys), np.uint32)
        for i in range(len(keys)):
            samp = summarize.sample(order[starts[i] : starts[i + 1]])
            j = summarize.pick_representative(xy64[samp], rep64[samp], tile_size)
            rep_ids[i] = samp[j]
        np.save(points_dir / f"z{z}_order.npy", order)
        np.save(points_dir / f"z{z}_keys.npy", keys.astype(np.uint32))
        np.save(points_dir / f"z{z}_starts.npy", starts)
        np.save(points_dir / f"z{z}_rep.npy", rep_ids)


# -------------------------------------------------------------- metadata

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")
_JOIN_HINTS = ("path", "filepath", "file", "filename", "image", "img", "name", "id")


def _sanitize_ident(name: str, taken: set[str]) -> str:
    ident = _IDENT_RE.sub("_", name.strip()) or "col"
    if ident[0].isdigit():
        ident = "_" + ident
    base, i = ident, 2
    while ident.lower() in taken:
        ident = f"{base}_{i}"
        i += 1
    taken.add(ident.lower())
    return ident


def _infer_type(values: list[str]):
    def all_parse(cast):
        ok = False
        for v in values:
            if v is None or v == "":
                continue
            try:
                cast(v)
                ok = True
            except ValueError:
                return False
        return ok

    if all_parse(int):
        return "INTEGER", int
    if all_parse(float):
        return "REAL", float
    return "TEXT", str


def load_metadata(meta_csv: Path, rels: list[str]):
    """Returns (columns: [(ident, sqltype)], rows: per-image value lists)."""
    with open(meta_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], None
    cols = list(rows[0].keys())

    by_alias: dict[str, dict[str, int]] = {}  # col -> alias -> row index
    for c in cols:
        amap = {}
        for ri, r in enumerate(rows):
            k = (r.get(c) or "").strip()
            if k:
                for alias in (k, Path(k).name, Path(k).stem):
                    amap.setdefault(alias, ri)
        by_alias[c] = amap

    def matches(c):
        amap = by_alias[c]
        return sum(
            1 for rel in rels if rel in amap or Path(rel).name in amap or Path(rel).stem in amap
        )

    candidates = sorted(cols, key=lambda c: (matches(c), c.lower() in _JOIN_HINTS), reverse=True)
    join_col = candidates[0]
    n_match = matches(join_col)
    if n_match == 0:
        raise SystemExit(
            f"metadata file {meta_csv}: no column matches any image path/filename "
            f"(columns: {cols})"
        )
    print(f"  metadata: join on column '{join_col}' ({n_match}/{len(rels)} images matched)")

    taken = {"id", "path", "width", "height"}
    out_cols, casts = [], []
    data_cols = [c for c in cols if c != join_col]
    for c in data_cols:
        sqltype, cast = _infer_type([r.get(c) for r in rows])
        out_cols.append((_sanitize_ident(c, taken), sqltype))
        casts.append(cast)

    amap = by_alias[join_col]
    per_image = []
    for rel in rels:
        ri = amap.get(rel)
        if ri is None:
            ri = amap.get(Path(rel).name)
        if ri is None:
            ri = amap.get(Path(rel).stem)
        if ri is None:
            per_image.append([None] * len(data_cols))
            continue
        vals = []
        for c, cast in zip(data_cols, casts):
            v = rows[ri].get(c)
            if v is None or v == "":
                vals.append(None)
            else:
                try:
                    vals.append(cast(v))
                except ValueError:
                    vals.append(v)
        per_image.append(vals)
    return out_cols, per_image


def build_db(db_path: Path, rels: list[str], sizes, user_cols, user_rows) -> None:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    col_sql = "".join(f', "{ident}" {sqltype}' for ident, sqltype in user_cols)
    con.execute(
        f"CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT, "
        f"width INTEGER, height INTEGER{col_sql})"
    )
    n_user = len(user_cols)
    placeholders = ",".join("?" * (4 + n_user))
    rows_iter = (
        (i, rel, sizes[i][0], sizes[i][1], *(user_rows[i] if user_rows else []))
        for i, rel in enumerate(rels)
    )
    con.executemany(f"INSERT INTO images VALUES ({placeholders})", rows_iter)
    # index low-cardinality user columns (cheap wins for tag-style filters)
    n = len(rels)
    for ident, _ in user_cols:
        (distinct,) = con.execute(f'SELECT COUNT(DISTINCT "{ident}") FROM images').fetchone()
        if distinct <= max(256, n // 100):
            con.execute(f'CREATE INDEX "idx_{ident}" ON images ("{ident}")')
    con.commit()
    con.close()


# ------------------------------------------------------------------ build

def run(args) -> None:
    images_root = Path(args.images).resolve()
    out_final = Path(args.out).resolve()
    if not images_root.is_dir():
        raise SystemExit(f"--images: not a directory: {images_root}")
    if out_final.exists() and any(out_final.iterdir()):
        if not args.force:
            raise SystemExit(f"output {out_final} exists and is not empty (use --force)")
        shutil.rmtree(out_final)

    if args.files:
        rels = sorted(
            line.strip()
            for line in Path(args.files).read_text().splitlines()
            if line.strip()
        )
        missing = [r for r in rels[:: max(1, len(rels) // 500)] if not (images_root / r).is_file()]
        if missing:
            raise SystemExit(
                f"--files: sampled paths not found under {images_root}, e.g. {missing[:3]}"
            )
    else:
        rels = find_images(images_root)
    if not rels:
        raise SystemExit(f"no images found under {images_root}")
    n = len(rels)
    print(f"Building dataset: {n} images -> {out_final}")

    out = out_final.parent / (out_final.name + ".building")
    if out.exists():
        shutil.rmtree(out)
    points_dir = out / "points"
    for d in (points_dir, out / "previews"):
        d.mkdir(parents=True)

    cell = args.cell

    # --- decode every image once; store raw sprites for O(1) access -----
    tasks = (
        (i, str(images_root / rel), str(out / preview_relpath(i)), cell, args.preview_max)
        for i, rel in enumerate(rels)
    )
    feats = np.empty((n, FEAT_SIZE * FEAT_SIZE * 3), np.float32)
    quality = np.empty(n, np.float32)
    sizes = [(0, 0)] * n
    sprites = np.memmap(out / "sprites.bin", dtype=np.uint8, mode="w+",
                        shape=(n, cell, cell, 3))

    workers = args.workers or os.cpu_count() or 1
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_process_one, tasks, chunksize=16)
    else:
        executor = None
        results = map(_process_one, tasks)

    try:
        for img_id, sprite_b, feat_b, w, h, q in results:
            sprites[img_id] = np.frombuffer(sprite_b, np.uint8).reshape(cell, cell, 3)
            feats[img_id] = np.frombuffer(feat_b, np.float32)
            quality[img_id] = q
            sizes[img_id] = (w, h)
            if (img_id + 1) % 1000 == 0 or img_id + 1 == n:
                print(f"  images: {img_id + 1}/{n}", end="\r", flush=True)
        print()
    finally:
        if executor:
            executor.shutdown()
        sprites.flush()
        # Windows won't rename/delete a file (or its directory) while a memory
        # map is open, so release the handle explicitly before the final rename.
        sprites._mmap.close()
        del sprites
        gc.collect()

    # --- coordinates ---------------------------------------------------
    if args.coords:
        xy = normalize_coords(load_coords(Path(args.coords), rels))
        coords_source = "provided"
    elif n >= 3 and np.isfinite(feats).all():
        xy = normalize_coords(pca_layout(feats.astype(np.float64)))
        coords_source = "visual-pca"
        print("  coords: no --coords given; using visual-PCA fallback layout")
    else:
        xy = normalize_coords(np.random.default_rng(0).random((n, 2)))
        coords_source = "random"
    np.save(points_dir / "xy.npy", xy)

    # --- scores ---------------------------------------------------------
    rep = (0.7 * density_scores(xy) + 0.3 * quality).astype(np.float32)
    np.save(points_dir / "rep.npy", rep)

    # --- tile indexes ----------------------------------------------------
    write_tile_indexes(points_dir, xy, rep, args.max_zoom)
    print(f"  tiles: z0..z{args.max_zoom} written")

    # --- metadata DB ------------------------------------------------------
    user_cols, user_rows = ([], None)
    if args.metadata:
        user_cols, user_rows = load_metadata(Path(args.metadata), rels)
    build_db(out / "metadata.sqlite", rels, sizes, user_cols, user_rows)

    # --- region labels + density underlay --------------------------------
    label_fields = labels.generate(out, args.label_column)

    # --- manifest last: marks the bundle complete -------------------------
    manifest = {
        "format_version": FORMAT_VERSION,
        "name": args.name or images_root.name,
        "count": n,
        "sprite_cell": cell,
        "zoom": {"min": 0, "max": args.max_zoom},
        "aggregate_threshold": args.threshold,
        "preview_max_side": args.preview_max,
        "coords_source": coords_source,
        "metadata_columns": [{"name": i, "type": t} for i, t in user_cols],
        **label_fields,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if out_final.exists():
        shutil.rmtree(out_final)
    out.rename(out_final)
    print(f"Done: {out_final}")


def add_build_options(p) -> None:
    """Build-stage options shared by `atlas build` and `atlas run`."""
    p.add_argument("--files", help="text file listing image paths relative to --images "
                                   "(one per line); skips directory scanning")
    p.add_argument("--metadata", help="CSV with a path/filename column + metadata columns")
    p.add_argument("--coords", help="CSV with path,x,y (precomputed 2D layout)")
    p.add_argument("--name", help="dataset display name")
    p.add_argument("--workers", type=int, default=0, help="decode workers (0 = cpu count)")
    p.add_argument("--cell", type=int, default=48, help="map sprite size in px")
    p.add_argument("--preview-max", type=int, default=512, help="detail preview max side")
    p.add_argument("--max-zoom", type=int, default=10, choices=range(1, 13))
    p.add_argument("--threshold", type=int, default=8, help="max items per tile before aggregating")
    p.add_argument("--label-column", help="metadata column to derive region labels from "
                                          "(a density underlay is always written)")


def add_parser(sub) -> None:
    p = sub.add_parser("build", help="build a dataset bundle from raw images")
    p.add_argument("--images", required=True, help="directory of images (recursive)")
    p.add_argument("--out", required=True, help="output dataset directory")
    add_build_options(p)
    p.add_argument("--force", action="store_true", help="overwrite existing output")
    p.set_defaults(func=run)
