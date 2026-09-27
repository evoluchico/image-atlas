"""Substantive (attitudinal) convergence measures on N x D stance matrices.

All distances use stance vectors normalised to [0, 1] per dimension.
Pairwise distances are divided by sqrt(D) so they also lie in [0, 1]
regardless of how many dimensions a topic defines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew

from ..config.schema import StanceDimension


def stance_matrices(
    probes: pd.DataFrame, dims: dict[str, StanceDimension]
) -> dict[int, pd.DataFrame]:
    """round -> DataFrame (index agent_id, columns dimensions, normalised to [0,1]).
    Agents with any missing value in a round are dropped for that round."""
    out: dict[int, pd.DataFrame] = {}
    if probes.empty:
        return out
    names = list(dims)
    lo = pd.Series({d: dims[d].min for d in names})
    span = pd.Series({d: dims[d].max - dims[d].min for d in names})
    for rnd, grp in probes.groupby("round"):
        wide = grp.pivot_table(index="agent_id", columns="stance_dimension", values="value", aggfunc="first")
        wide = wide.reindex(columns=names)
        wide = ((wide - lo) / span).dropna()
        out[int(rnd)] = wide.sort_index()
    return out


def mean_pairwise_distance(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    return float(pdist(mat, "euclidean").mean() / np.sqrt(mat.shape[1]))


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient; > 5/9 (~0.555) suggests bi-/multimodality."""
    n = len(x)
    if n < 4 or np.std(x) == 0:
        return float("nan")
    g = skew(x, bias=False)
    k = kurtosis(x, bias=False)
    return float((g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def dispersion_rows(mat: pd.DataFrame) -> dict[str, float]:
    rows: dict[str, float] = {}
    for d in mat.columns:
        x = mat[d].to_numpy()
        rows[f"stance_sd:{d}"] = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
        rows[f"stance_mad:{d}"] = float(np.median(np.abs(x - np.median(x))))
        q75, q25 = np.percentile(x, [75, 25])
        rows[f"stance_iqr:{d}"] = float(q75 - q25)
        rows[f"stance_mean:{d}"] = float(np.mean(x))
        rows[f"bimodality:{d}"] = bimodality_coefficient(x)
    rows["stance_sd_mean"] = float(np.nanmean([rows[f"stance_sd:{d}"] for d in mat.columns]))
    return rows


def cluster_summary(mat: np.ndarray, max_k: int, seed: int) -> dict[str, float]:
    """Descriptive clustering diagnostic: Gaussian mixtures with k = 1..max_k
    chosen by BIC. Nothing forces two camps; k = 1 is a valid answer."""
    from sklearn.mixture import GaussianMixture

    n = mat.shape[0]
    kmax = int(max(1, min(max_k, n // 3)))
    if n < 4:
        return {"cluster_count": float("nan")}
    best = None
    for k in range(1, kmax + 1):
        gm = GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=seed, n_init=2).fit(mat)
        bic = gm.bic(mat)
        if best is None or bic < best[0]:
            best = (bic, k, gm)
    _, k, gm = best
    labels = gm.predict(mat)
    within = []
    for c in np.unique(labels):
        pts = mat[labels == c]
        if len(pts) > 1:
            within.append(pdist(pts).mean() / np.sqrt(mat.shape[1]))
    centres = np.array([mat[labels == c].mean(axis=0) for c in np.unique(labels)])
    between = pdist(centres).mean() / np.sqrt(mat.shape[1]) if len(centres) > 1 else 0.0
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return {
        "cluster_count": float(len(sizes)),
        "cluster_within_distance": float(np.mean(within)) if within else 0.0,
        "cluster_between_distance": float(between),
        "cluster_minority_share": float(sizes.min() / n) if len(sizes) > 1 else 0.0,
    }
