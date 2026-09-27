"""Load and fingerprint experiment configuration files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .schema import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return ExperimentConfig.model_validate(data)


def config_from_dict(data: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(data)


def canonical_json(cfg: ExperimentConfig) -> str:
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def config_hash(cfg: ExperimentConfig) -> str:
    """Stable fingerprint of the *resolved* configuration (defaults included)."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def dump_config_yaml(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
