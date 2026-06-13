"""Generate a synthetic image collection and build a dataset from it.

This is the smoke test and first-run experience: six visual families with
distinct palettes, a metadata CSV, then a normal `atlas build` run (which
exercises the visual-PCA fallback layout — the families form clusters).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from . import build

SIZE = 256

PALETTES = {
    "rings": [(220, 60, 50), (245, 130, 50), (250, 200, 80)],
    "stripes": [(40, 80, 200), (80, 160, 230), (180, 220, 250)],
    "blobs": [(30, 130, 70), (90, 190, 110), (200, 240, 180)],
    "checker": [(120, 50, 160), (190, 110, 220), (240, 200, 250)],
    "noise": [(60, 60, 60), (140, 140, 140), (220, 220, 220)],
    "gradient": [(220, 180, 40), (40, 170, 160), (250, 240, 200)],
}


def _grid():
    ax = np.linspace(-1, 1, SIZE)
    return np.meshgrid(ax, ax)


def gen_image(rng: np.random.Generator, family: str) -> np.ndarray:
    c = [np.array(col, float) for col in PALETTES[family]]
    xx, yy = _grid()
    if family == "rings":
        r = np.sqrt(xx**2 + yy**2) * rng.uniform(3, 8) + rng.uniform(0, 3)
        t = (np.sin(r * np.pi) + 1) / 2
    elif family == "stripes":
        ang = rng.uniform(0, np.pi)
        t = (np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * rng.uniform(5, 15)) + 1) / 2
    elif family == "blobs":
        t = np.zeros_like(xx)
        for _ in range(rng.integers(3, 8)):
            cx, cy = rng.uniform(-1, 1, 2)
            t += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / rng.uniform(0.02, 0.15))
        t = np.clip(t, 0, 1)
    elif family == "checker":
        n = rng.integers(3, 10)
        t = ((np.floor((xx + 1) * n / 2) + np.floor((yy + 1) * n / 2)) % 2)
    elif family == "noise":
        small = rng.random((rng.integers(4, 16),) * 2)
        t = np.asarray(
            Image.fromarray((small * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.NEAREST),
            float,
        ) / 255.0
    else:  # gradient
        ang = rng.uniform(0, 2 * np.pi)
        t = (xx * np.cos(ang) + yy * np.sin(ang) + 2) / 4
    mix = rng.uniform(0.2, 0.8)
    img = c[0] * (1 - t)[..., None] + c[1] * t[..., None]
    img = img * (1 - mix * 0.3) + c[2] * (t**2)[..., None] * mix * 0.3
    return np.clip(img, 0, 255).astype(np.uint8)


def run(args) -> None:
    out = Path(args.out).resolve()
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    families = list(PALETTES)
    rng = np.random.default_rng(42)

    rows = []
    print(f"Generating {args.count} synthetic images in {images_dir}")
    for i in range(args.count):
        family = families[i % len(families)]
        px = gen_image(rng, family)
        name = f"{family}_{i:05d}.png"
        Image.fromarray(px).save(images_dir / name)
        rows.append(
            {
                "filename": name,
                "family": family,
                "year": int(rng.integers(1990, 2026)),
                "brightness": round(float(px.mean()) / 255.0, 3),
            }
        )
        if (i + 1) % 200 == 0 or i + 1 == args.count:
            print(f"  {i + 1}/{args.count}", end="\r", flush=True)
    print()

    meta_csv = out / "metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    build.run(
        SimpleNamespace(
            images=str(images_dir),
            files=None,
            out=str(out / "dataset"),
            metadata=str(meta_csv),
            coords=None,
            name="demo collection",
            workers=args.workers,
            cell=48,
            preview_max=512,
            max_zoom=8,
            threshold=8,
            force=True,
        )
    )
    print(f"\nNow run:  python -m atlas serve {out / 'dataset'}")


def add_parser(sub) -> None:
    p = sub.add_parser("demo", help="generate a synthetic dataset (smoke test)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(func=run)
