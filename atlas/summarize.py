"""Representative-selection heuristic (DESIGN.md section 6).

Shared by the build stage (precomputed unfiltered per-tile representatives)
and the server (filtered tiles at query time) so the two can never drift.
"""
from __future__ import annotations

import numpy as np

LAMBDA = 0.25      # distance penalty weight
SAMPLE_CAP = 2048  # max points used for center/representative math per tile


def sample(ids: np.ndarray) -> np.ndarray:
    """Deterministic stratified subsample; counts elsewhere stay exact."""
    n = len(ids)
    if n <= SAMPLE_CAP:
        return ids
    return ids[np.linspace(0, n - 1, SAMPLE_CAP).astype(np.intp)]


def pick_representative(pos: np.ndarray, w: np.ndarray, tile_size: float) -> int:
    """weighted center -> snap to a real point -> argmax(rep - λ·dist).

    pos (M,2) float64, w (M,) float64. Returns an index into pos/w.
    """
    wsum = w.sum()
    if wsum <= 0:
        w = np.ones(len(pos))
        wsum = float(len(pos))
    center = (pos * w[:, None]).sum(axis=0) / wsum
    anchor = pos[int(np.argmin(((pos - center) ** 2).sum(axis=1)))]
    dist = np.sqrt(((pos - anchor) ** 2).sum(axis=1)) / tile_size
    return int(np.argmax(w - LAMBDA * dist))
