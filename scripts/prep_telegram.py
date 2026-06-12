"""Prepare atlas build inputs from the gui_popolin Telegram concept-pipeline outputs.

Reads the aligned embeddings + paths produced by the concept pipeline,
computes a 2D projection (UMAP if available, else pure-numpy PCA), and
joins concept/topic labels. Writes into --out:

  files.txt      paths relative to --images-root (only valid images)
  coords.csv     filename,x,y
  metadata.csv   filename,year,month,day,hour,concept,topic

Run with an env that has umap-learn for the good layout, e.g.:
  ~/miniforge3/envs/atlas-umap/bin/python scripts/prep_telegram.py --out ... [--year 2020]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_BASE = Path("/media/chico/data_cclab/gui_popolin")
DEFAULT_CONCEPT = DEFAULT_BASE / "concept" / "output_images_2020_2023_messages"
DEFAULT_IMAGES_ROOT = DEFAULT_BASE / "CSN" / "build" / "images_telegram"

DATE_RE = re.compile(r"@(\d{2})-(\d{2})-(\d{4})_(\d{2})-\d{2}-\d{2}")


def project_umap(emb: np.ndarray, seed: int | None) -> np.ndarray:
    import umap

    print(f"  UMAP on {emb.shape} (cosine, n_neighbors=15, seed={seed})...", flush=True)
    # a fixed random_state forces single-threaded UMAP; omit it for big runs
    kw = {"random_state": seed} if seed is not None else {}
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
        verbose=True, **kw,
    )
    return reducer.fit_transform(emb)


def project_pca(emb: np.ndarray) -> np.ndarray:
    print(f"  PCA on {emb.shape} (umap unavailable fallback)...", flush=True)
    X = emb - emb.mean(axis=0)
    cov = (X.T @ X) / len(X)
    _, vecs = np.linalg.eigh(cov)
    return X @ vecs[:, -2:][:, ::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-dir", default=str(DEFAULT_CONCEPT))
    ap.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", help="restrict to one year subfolder (e.g. 2020)")
    ap.add_argument("--method", choices=["umap", "pca"], default="umap")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the UMAP seed (disables UMAP parallelism)")
    args = ap.parse_args()

    cdir = Path(args.concept_dir)
    images_root = Path(args.images_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = json.load(open(cdir / "2020_2021_2022_2023_all_image_paths.json"))
    rels = [str(Path(p).relative_to(images_root)) for p in paths]
    print(f"valid images: {len(rels)}")

    idx = np.arange(len(rels))
    if args.year:
        idx = np.array([i for i in idx if rels[i].startswith(args.year + "/")])
        print(f"subset {args.year}: {len(idx)}")
        if not len(idx):
            sys.exit(f"no images under year {args.year}")

    # concept id per image (csv is aligned with paths; verify by path anyway)
    concept_by_path: dict[str, str] = {}
    with open(cdir / "image_concepts.csv", newline="") as f:
        for row in csv.DictReader(f):
            concept_by_path[row["image_path"]] = row["concept"]

    # concept id -> short topic label (first 3 keywords)
    topic_label: dict[str, str] = {}
    with open(cdir / "concept_topics_clean_pt.csv", newline="") as f:
        for row in csv.DictReader(f):
            kws = [k.strip() for k in row["topics_clean_pt"].split(",")][:3]
            topic_label[row["concept"]] = ", ".join(kws)

    emb = np.load(cdir / "2020_2021_2022_2023_all_image_embeddings.npy", mmap_mode="r")
    assert len(emb) == len(rels), (len(emb), len(rels))
    sub = np.asarray(emb[idx], dtype=np.float32)

    if args.method == "umap":
        try:
            xy = project_umap(sub, args.seed)
        except ImportError:
            xy = project_pca(sub)
    else:
        xy = project_pca(sub)

    with open(out / "files.txt", "w") as f:
        f.writelines(rels[i] + "\n" for i in idx)

    with open(out / "coords.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "x", "y"])
        for j, i in enumerate(idx):
            w.writerow([rels[i], f"{xy[j, 0]:.5f}", f"{xy[j, 1]:.5f}"])

    n_meta = 0
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "year", "month", "day", "hour", "concept", "topic"])
        for i in idx:
            m = DATE_RE.search(rels[i])
            day, month, year, hour = m.groups() if m else ("", "", "", "")
            cid = concept_by_path.get(paths[i], "")
            w.writerow([rels[i], year, month, day, hour, cid, topic_label.get(cid, "")])
            n_meta += 1 if cid else 0
    print(f"wrote {out}/files.txt, coords.csv, metadata.csv ({n_meta} with concept ids)")


if __name__ == "__main__":
    main()
