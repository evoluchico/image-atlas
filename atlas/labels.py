"""Region labels + density underlay for a dataset bundle.

Both derive only from stored coordinates and a metadata column, so they can
be (re)computed for an existing bundle in seconds — no image re-decoding.

Labels are multi-scale: each distinct value of the chosen column becomes one
label placed at the spatial median of its images, with a `level` assigned by
magnitude (the biggest themes are level 0 and show when zoomed out; smaller
ones get higher levels and appear as you zoom in). The frontend declutters by
screen-space collision, so you see the "top features in view" at every zoom —
high-level regions that break into finer ones, like a thematic map.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

DENSITY_RES = 320
LEVEL_BASE = 2.5      # each ~2.5x drop in count = one finer label level
MAX_LEVEL = 6


def _blur(a: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = sum(p[1 + dy: a.shape[0] + 1 + dy, 1 + dx: a.shape[1] + 1 + dx]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return a


def _ramp(t: np.ndarray) -> np.ndarray:
    """Map intensity [0,1] to a dark-navy -> magenta -> pink RGBA glow."""
    stops = np.array([
        [0.00, 20, 16, 48],
        [0.35, 90, 28, 120],
        [0.70, 210, 48, 150],
        [1.00, 255, 190, 225],
    ])
    rgb = np.stack([np.interp(t, stops[:, 0], stops[:, i]) for i in (1, 2, 3)], axis=-1)
    alpha = np.clip(t * 1.5, 0, 1) ** 0.85 * 255
    return np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)


def _density_field(xy: np.ndarray) -> np.ndarray:
    """Smoothed, log-scaled point-density on a [0,1] grid, normalized to [0,1]."""
    hist, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=DENSITY_RES, range=[[0, 1], [0, 1]]
    )
    t = np.log1p(_blur(hist, passes=3))
    return t / (t.max() + 1e-9)


def build_density(field: np.ndarray, out_path: Path) -> None:
    Image.fromarray(_ramp(field), "RGBA").save(out_path, "WEBP", quality=80)


# marching-squares case table: case -> list of (edgeA, edgeB) line segments.
# corner bits: TL=1, TR=2, BR=4, BL=8.  edges: T(op) R(ight) B(ottom) L(eft).
_MS = {
    1: [("L", "T")], 2: [("T", "R")], 3: [("L", "R")], 4: [("R", "B")],
    5: [("L", "T"), ("R", "B")], 6: [("T", "B")], 7: [("L", "B")],
    8: [("B", "L")], 9: [("T", "B")], 10: [("T", "R"), ("B", "L")],
    11: [("R", "B")], 12: [("L", "R")], 13: [("T", "R")], 14: [("L", "T")],
}
CONTOUR_LEVELS = [0.18, 0.32, 0.47, 0.63, 0.80]


def build_contours(field: np.ndarray, out_path: Path) -> int:
    R = field.shape[0]
    A, B = field[:-1, :-1], field[:-1, 1:]
    D, E = field[1:, :-1], field[1:, 1:]
    levels = []
    total = 0
    for lv in CONTOUR_LEVELS:
        case = ((A > lv) * 1 + (B > lv) * 2 + (E > lv) * 4 + (D > lv) * 8)
        rows, cols = np.nonzero((case != 0) & (case != 15))
        segs = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            a, b, d, e = field[r, c], field[r, c + 1], field[r + 1, c], field[r + 1, c + 1]

            def pt(edge):
                if edge == "T":
                    t = (lv - a) / (b - a) if b != a else 0.5
                    return ((c + 0.5 + t) / R, (r + 0.5) / R)
                if edge == "B":
                    t = (lv - d) / (e - d) if e != d else 0.5
                    return ((c + 0.5 + t) / R, (r + 1.5) / R)
                if edge == "L":
                    t = (lv - a) / (d - a) if d != a else 0.5
                    return ((c + 0.5) / R, (r + 0.5 + t) / R)
                t = (lv - b) / (e - b) if e != b else 0.5  # "R"
                return ((c + 1.5) / R, (r + 0.5 + t) / R)

            for e1, e2 in _MS[int(case[r, c])]:
                (x0, y0), (x1, y1) = pt(e1), pt(e2)
                segs += [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]
        levels.append({"t": lv, "segments": segs})
        total += len(segs) // 4
    out_path.write_text(json.dumps({"levels": levels}), encoding="utf-8")
    return total


def build_labels(xy: np.ndarray, values: list, out_path: Path, max_labels: int) -> int:
    # Group by the first comma-separated keyword, so verbose multi-keyword
    # topics that share a lead term ("news brasil, cnn..." / "news brasil,
    # noticias...") collapse into one clean region name ("news brasil"). For
    # atomic columns (a plain category) the key is the whole value — no merge.
    groups: dict[str, list[int]] = {}
    for i, v in enumerate(values):
        if v is None:
            continue
        key = str(v).split(",")[0].strip()
        if key:
            groups.setdefault(key, []).append(i)
    if not groups:
        out_path.write_text(json.dumps({"labels": []}), encoding="utf-8")
        return 0

    n = len(values)
    min_count = max(3, n // 5000)
    items = []
    for text, ids in groups.items():
        if len(ids) < min_count:
            continue
        pts = xy[ids]
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        items.append((text, cx, cy, len(ids)))
    if not items:
        out_path.write_text(json.dumps({"labels": []}), encoding="utf-8")
        return 0

    items.sort(key=lambda t: -t[3])
    items = items[:max_labels]
    maxc = items[0][3]
    labels = [
        {
            "text": text,
            "x": round(cx, 5),
            "y": round(cy, 5),
            "count": cnt,
            "level": min(MAX_LEVEL, int(np.log(maxc / cnt) / np.log(LEVEL_BASE))),
        }
        for text, cx, cy, cnt in items
    ]
    out_path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return len(labels)


def generate(dataset: Path, column: str | None, max_labels: int = 800) -> dict:
    """Compute density.webp (always) and labels.json (if a column is given).
    Returns manifest fields to merge: {"has_density": True, "labels_column": ...}."""
    xy = np.load(dataset / "points" / "xy.npy")
    field = _density_field(xy)
    build_density(field, dataset / "density.webp")
    n_seg = build_contours(field, dataset / "density_contours.json")
    out = {"has_density": True}

    if column:
        con = sqlite3.connect(f"file:{dataset / 'metadata.sqlite'}?mode=ro", uri=True)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(images)")]
            if column not in cols:
                raise SystemExit(
                    f"--label-column {column!r} not in metadata (have: "
                    f"{[c for c in cols if c not in ('id','path','width','height')]})"
                )
            rows = con.execute(
                f'SELECT id, "{column}" FROM images ORDER BY id'
            ).fetchall()
        finally:
            con.close()
        values = [v for _, v in rows]
        k = build_labels(xy, values, dataset / "labels.json", max_labels)
        out["labels_column"] = column
        print(f"  labels: {k} from column '{column}'")
    print(f"  density underlay + {n_seg} contour segments written")
    return out


def run(args) -> None:
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{dataset} is not a dataset bundle (no manifest.json)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"labelling '{manifest['name']}' ({manifest['count']} images)")
    manifest.update(generate(dataset, args.column, args.max_labels))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("done")


def add_parser(sub) -> None:
    p = sub.add_parser("label", help="(re)compute region labels + density for a bundle")
    p.add_argument("dataset", help="dataset bundle directory")
    p.add_argument("--column", help="metadata column to label regions by "
                                    "(omit to write only the density underlay)")
    p.add_argument("--max-labels", type=int, default=800)
    p.set_defaults(func=run)
