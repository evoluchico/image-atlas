"""Statistics across runs.

The simulation *run* is the unit of replication: individual messages or
agents within a run are never treated as independent observations. Trajectory
summaries bootstrap over runs; paired comparisons use the fact that
environments of the same replicate share an initial population.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

FACTORS = ["environment_type", "population_size"]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def condition_trajectories(metrics: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Mean and run-bootstrap CI per environment x N x round x metric."""
    rows = []
    for key, grp in metrics.groupby([*FACTORS, "round", "metric_name"]):
        vals = grp["metric_value"].to_numpy(float)
        lo, hi = bootstrap_ci(vals, rng, n_boot)
        rows.append(
            dict(zip([*FACTORS, "round", "metric_name"], key))
            | {"mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
               "sd": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan"),
               "ci_low": lo, "ci_high": hi, "n_runs": int(np.isfinite(vals).sum())}
        )
    return pd.DataFrame(rows)


def final_values(metrics: pd.DataFrame) -> pd.DataFrame:
    """Each run's value at its last observed round, per metric."""
    idx = metrics.groupby(["run_id", "metric_name"])["round"].idxmax()
    return metrics.loc[idx]


def paired_differences(
    metrics: pd.DataFrame, metric_names: list[str], rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    """Final-round differences between environments within the same replicate
    and population size (same initial population), bootstrapped over pairs."""
    fin = final_values(metrics[metrics["metric_name"].isin(metric_names)])
    rows = []
    envs = sorted(fin["environment_type"].unique())
    for (name, n), grp in fin.groupby(["metric_name", "population_size"]):
        wide = grp.pivot_table(index="replicate", columns="environment_type", values="metric_value")
        for a, b in itertools.combinations(envs, 2):
            if a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).dropna().to_numpy()
            if len(d) == 0:
                continue
            lo, hi = bootstrap_ci(d, rng, n_boot)
            rows.append(
                {"metric_name": name, "population_size": n, "environment_a": a, "environment_b": b,
                 "mean_difference": float(d.mean()), "ci_low": lo, "ci_high": hi, "n_pairs": len(d)}
            )
    return pd.DataFrame(rows)


def mixed_models(metrics: pd.DataFrame, metric_names: list[str]) -> str:
    """outcome ~ environment * N + round with a random intercept per run.
    Requires statsmodels (optional). Returns a text report."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return "statsmodels not installed: `pip install llm-society[stats]` for mixed-effects models.\n"
    out = []
    for name in metric_names:
        df = metrics[(metrics["metric_name"] == name) & (metrics["round"] > 0)].dropna(subset=["metric_value"])
        if df["run_id"].nunique() < 4 or df["environment_type"].nunique() < 2:
            continue
        df = df.assign(N=df["population_size"].astype(str))
        formula = "metric_value ~ C(environment_type) * C(N) + round" if df["N"].nunique() > 1 else "metric_value ~ C(environment_type) + round"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formula, df, groups=df["run_id"]).fit()
            out.append(f"=== {name}\n{res.summary()}\n")
        except Exception as exc:  # singular fits etc. are reported, not fatal
            out.append(f"=== {name}\nmodel failed: {exc}\n")
    return "\n".join(out) or "not enough runs for mixed models\n"
