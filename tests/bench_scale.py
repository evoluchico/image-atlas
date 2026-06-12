"""Scale benchmark: fabricate a 1M-point bundle (no real images) and measure
the runtime hot paths against the budget in DESIGN.md section 9.

Run:  python tests/bench_scale.py [--n 1000000] [--keep]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas import FORMAT_VERSION, build  # noqa: E402
from atlas.server import Dataset  # noqa: E402

FAMILIES = [f"family_{i}" for i in range(10)]


def fabricate(root: Path, n: int, max_zoom: int = 8) -> None:
    print(f"fabricating bundle with {n:,} points at {root}")
    rng = np.random.default_rng(0)
    points = root / "points"
    points.mkdir(parents=True)
    (root / "previews").mkdir()

    # clustered layout: mixture of gaussians (realistic tile skew)
    k = 40
    centers = rng.random((k, 2))
    which = rng.integers(0, k, n)
    xy = centers[which] + rng.normal(0, 0.035, (n, 2))
    xy = np.clip(xy, 0.0, 1.0).astype(np.float32)
    rep = rng.random(n).astype(np.float32)
    np.save(points / "xy.npy", xy)
    np.save(points / "rep.npy", rep)

    t0 = time.perf_counter()
    build.write_tile_indexes(points, xy, rep, max_zoom)
    print(f"  tile indexes + reps z0..z{max_zoom}: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    con = sqlite3.connect(root / "metadata.sqlite")
    con.execute(
        "CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT, width INTEGER,"
        " height INTEGER, family TEXT, year INTEGER, score REAL)"
    )
    fam = rng.integers(0, len(FAMILIES), n)
    year = rng.integers(1900, 2026, n)
    score = rng.random(n).round(4)
    con.executemany(
        "INSERT INTO images VALUES (?,?,?,?,?,?,?)",
        (
            (i, f"img_{i:07d}.jpg", 800, 600, FAMILIES[fam[i]], int(year[i]), float(score[i]))
            for i in range(n)
        ),
    )
    con.execute("CREATE INDEX idx_family ON images (family)")
    con.commit()
    con.close()
    print(f"  sqlite ({n:,} rows): {time.perf_counter() - t0:.1f}s")

    cell = 8  # small sprites keep the fabricated store at ~192 MB for 1M
    sprites = np.memmap(root / "sprites.bin", dtype=np.uint8, mode="w+",
                        shape=(n, cell, cell, 3))
    sprites.flush()
    del sprites
    (root / "manifest.json").write_text(json.dumps({
        "format_version": FORMAT_VERSION, "name": "bench", "count": n,
        "sprite_cell": cell,
        "zoom": {"min": 0, "max": max_zoom}, "aggregate_threshold": 8,
        "preview_max_side": 512, "coords_source": "random",
        "metadata_columns": [{"name": "family", "type": "TEXT"},
                              {"name": "year", "type": "INTEGER"},
                              {"name": "score", "type": "REAL"}],
    }))


def timed(label, fn, repeat=5):
    fn()  # warm
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  {label:<58} {min(times):8.1f} ms (best of {repeat})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--keep", action="store_true", help="keep the fabricated bundle")
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="atlas-bench-"))
    try:
        fabricate(root, args.n)

        t0 = time.perf_counter()
        ds = Dataset(root)
        print(f"\nDataset open + validate: {(time.perf_counter() - t0) * 1000:.0f} ms")

        print("\nhot paths (budget: filter <= ~150 ms cold, viewport <= ~10 ms):")
        t0 = time.perf_counter()
        token, count = ds.make_filter("family = 'family_3' AND year >= 1990")
        print(f"  {'filter cold (indexed col + range), ' + format(count, ','):<58}"
              f" {(time.perf_counter() - t0) * 1000:8.1f} ms")
        t0 = time.perf_counter()
        ds.make_filter("score < 0.31")
        print(f"  {'filter cold (full scan on REAL)':<58}"
              f" {(time.perf_counter() - t0) * 1000:8.1f} ms")
        timed("filter warm (cache hit)", lambda: ds.make_filter("score < 0.31"))

        # z4 full world = 256 tiles: the realistic zoomed-out worst case
        timed("viewport z4 full world, no filter",
              lambda: ds.viewport(4, 0, 0, 1, 1, "all"))
        timed("viewport z4 full world, filtered",
              lambda: ds.viewport(4, 0, 0, 1, 1, token))
        timed("viewport z0 single tile (all 1M points), no filter",
              lambda: ds.viewport(0, 0, 0, 1, 1, "all"))
        timed("viewport z8 quarter view, filtered",
              lambda: ds.viewport(8, 0.4, 0.4, 0.65, 0.65, token))

        strip_ids = list(range(0, args.n, max(1, args.n // 250)))[:250]
        timed("sprite strip (250 sprites, encode)",
              lambda: (ds._strips.clear(), ds.sprite_strip(strip_ids)))

        circle = [[0.5 + 0.18 * np.cos(a), 0.5 + 0.18 * np.sin(a)]
                  for a in np.linspace(0, 2 * np.pi, 64)]
        t0 = time.perf_counter()
        stok, scount = ds.make_selection(circle, token)
        print(f"  {'lasso select (64-vertex circle, on filter), ' + format(scount, ','):<58}"
              f" {(time.perf_counter() - t0) * 1000:8.1f} ms")

        r = ds.viewport(4, 0, 0, 1, 1, stok)
        total = sum(a["count"] for a in r["aggregates"]) + len(r["items"])
        assert total == scount, f"conservation failed: {total} != {scount}"
        print(f"\nconservation check at z4: {total:,} == {scount:,}  OK")
    finally:
        if args.keep:
            print(f"kept: {root}")
        else:
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
