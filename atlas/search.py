"""Semantic (CLIP text->image) search over a dataset bundle.

`atlas embed` stores precomputed image embeddings into the bundle (mapped to
bundle ids by path, L2-normalized, float16). At serve time a TextSearcher
lazy-loads the matching CLIP text encoder, encodes the query, and ranks images
by cosine similarity — brute force, which is a few milliseconds at ~1M.

The text encoder must be the model that produced the image embeddings, or the
two live in different spaces; the model name is recorded in the manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "clip-ViT-B-32-multilingual-v1"


def _path_keys(p: str):
    name = Path(p).name
    return (p, name, Path(name).stem)


def store_embeddings(dataset: Path, emb_path: Path, paths_json: Path,
                     model: str, images_root: Path | None) -> dict:
    import sqlite3

    emb = np.load(emb_path, mmap_mode="r")
    src_paths = json.loads(Path(paths_json).read_text(encoding="utf-8"))
    if len(src_paths) != len(emb):
        raise SystemExit(f"paths ({len(src_paths)}) != embeddings ({len(emb)})")

    # map every alias of each source path -> its embedding row
    row_of: dict[str, int] = {}
    root = str(images_root) if images_root else None
    for i, p in enumerate(src_paths):
        rel = p
        if root and p.startswith(root):
            rel = p[len(root):].lstrip("/\\")
        for k in (*_path_keys(p), *_path_keys(rel)):
            row_of.setdefault(k, i)

    con = sqlite3.connect(f"file:{dataset / 'metadata.sqlite'}?mode=ro", uri=True)
    bundle_paths = [r[0] for r in con.execute("SELECT path FROM images ORDER BY id")]
    con.close()

    dim = emb.shape[1]
    out = np.zeros((len(bundle_paths), dim), np.float32)
    missing = 0
    for i, bp in enumerate(bundle_paths):
        row = next((row_of[k] for k in _path_keys(bp) if k in row_of), None)
        if row is None:
            missing += 1
            continue
        out[i] = emb[row]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.where(norm > 0, norm, 1.0)

    (dataset / "search").mkdir(exist_ok=True)
    np.save(dataset / "search" / "embeddings.npy", out.astype(np.float16))
    print(f"  embeddings: {len(bundle_paths) - missing}/{len(bundle_paths)} matched"
          + (f" ({missing} missing -> never match)" if missing else ""))
    return {"search": {"model": model, "dim": int(dim)}}


class TextSearcher:
    """Lazy CLIP text->image search held by the server."""

    def __init__(self, dataset: Path, model: str):
        self._emb_path = dataset / "search" / "embeddings.npy"
        self._model_name = model
        self._emb = None      # (N, dim) float32, loaded on first query
        self._model = None

    def _ensure(self):
        if self._emb is None:
            self._emb = np.load(self._emb_path).astype(np.float32)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def ensure_matrix(self) -> np.ndarray:
        """Load and return the (N, dim) image embedding matrix (no model)."""
        if self._emb is None:
            self._emb = np.load(self._emb_path).astype(np.float32)
        return self._emb

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode text(s) into the shared CLIP space, L2-normalized."""
        self._ensure()
        return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def rank(self, query: str) -> np.ndarray:
        """Return image ids sorted by descending similarity to the query text."""
        self._ensure()
        q = self._model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims = self._emb @ q
        return np.argsort(-sims, kind="stable").astype(np.int64), sims


def run_embed(args) -> None:
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{dataset} is not a dataset bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"embedding '{manifest['name']}' ({manifest['count']} images)")
    fields = store_embeddings(
        dataset, Path(args.embeddings), Path(args.paths), args.model,
        Path(args.images_root) if args.images_root else None,
    )
    manifest.update(fields)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("done")


def add_parser(sub) -> None:
    p = sub.add_parser("embed", help="store image embeddings for CLIP text search")
    p.add_argument("dataset", help="dataset bundle directory")
    p.add_argument("--embeddings", required=True, help="(N, dim) .npy of image embeddings")
    p.add_argument("--paths", required=True, help="JSON list of image paths aligned to --embeddings")
    p.add_argument("--images-root", help="prefix to strip from --paths to match bundle paths")
    p.add_argument("--model", default=DEFAULT_MODEL, help="CLIP model that made the embeddings")
    p.set_defaults(func=run_embed)
