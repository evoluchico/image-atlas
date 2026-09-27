"""Deterministic seed derivation.

Every stochastic process gets its own generator derived from the master seed
and a tuple of labels, so e.g. the routing stream of one environment can
never perturb the population stream shared across environments.

    master_seed
        +-- ("profile", N, replicate)        -> initial population (shared by all environments)
        +-- ("network", N, replicate)        -> Erdős–Rényi graph
        +-- ("routing", N, replicate, env)   -> recipient selection, exposure sampling
        +-- ("ordering", N, replicate, env)  -> forum post ordering
        +-- ("analysis", ...)                -> bootstrap, clustering, topic models
"""

from __future__ import annotations

import hashlib

import numpy as np


def _label_int(label: object) -> int:
    return int.from_bytes(hashlib.sha256(str(label).encode("utf-8")).digest()[:4], "little")


def derive_seed(master_seed: int, *labels: object) -> int:
    ss = np.random.SeedSequence(entropy=int(master_seed), spawn_key=tuple(_label_int(l) for l in labels))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def make_rng(master_seed: int, *labels: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(master_seed, *labels))


def rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def restore_rng(state: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng
