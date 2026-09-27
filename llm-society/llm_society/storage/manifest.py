"""Experiment metadata manifest (reproducibility record)."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from .. import __version__
from .parquet import atomic_write_json

PACKAGES = [
    "llm-society", "pydantic", "PyYAML", "numpy", "scipy", "pandas", "pyarrow", "networkx",
    "scikit-learn", "matplotlib", "httpx", "anthropic", "sentence-transformers", "statsmodels",
]


def git_info(start: Path | None = None) -> dict:
    cwd = start or Path(__file__).resolve().parent
    def git(*args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    try:
        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None}
        dirty = git("status", "--porcelain")
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None}


def package_versions() -> dict[str, str | None]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    out["llm_society(module)"] = __version__
    return out


def write_manifest(exp_dir: Path, **fields) -> dict:
    manifest = {
        "git": git_info(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        **fields,
    }
    atomic_write_json(exp_dir / "manifest.json", manifest)
    return manifest
