"""Recompute the tile indexes of an existing bundle without rebuilding it.

Tile indexes (and per-tile representatives) derive purely from the stored
coordinates and rep scores, so changing the zoom depth or the aggregate
threshold doesn't require re-decoding any images — seconds instead of the
full build.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import FORMAT_VERSION, build


def run(args) -> None:
    root = Path(args.dataset).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{root} is not a dataset bundle (no manifest.json)")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise SystemExit(
            f"unsupported format_version {manifest.get('format_version')!r}; re-run `atlas build`"
        )

    points = root / "points"
    xy = np.load(points / "xy.npy")
    rep = np.load(points / "rep.npy")
    old_max = manifest["zoom"]["max"]
    print(f"retiling '{manifest['name']}' ({len(xy)} images): "
          f"z0..z{old_max} -> z0..z{args.max_zoom}")

    build.write_tile_indexes(points, xy, rep, args.max_zoom)

    manifest["zoom"] = {"min": 0, "max": args.max_zoom}
    if args.threshold is not None:
        manifest["aggregate_threshold"] = args.threshold
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # stale deeper levels from a previous, deeper tiling
    for z in range(args.max_zoom + 1, 16):
        for kind in ("order", "keys", "starts", "rep"):
            (points / f"z{z}_{kind}.npy").unlink(missing_ok=True)
    print("done")


def add_parser(sub) -> None:
    p = sub.add_parser("retile", help="change zoom depth / threshold of an existing bundle")
    p.add_argument("dataset", help="dataset bundle directory")
    p.add_argument("--max-zoom", type=int, default=11, choices=range(1, 13))
    p.add_argument("--threshold", type=int, default=None,
                   help="also change images-per-tile before aggregating")
    p.set_defaults(func=run)
