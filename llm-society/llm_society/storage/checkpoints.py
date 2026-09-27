"""Round-level checkpoints and the within-round response cache.

A checkpoint is written after every completed round (including its probe):
it holds the full run state (environment incl. RNG state, memories, all
records so far). A crash therefore costs at most the round in progress, and
even that is mostly recovered: every completed LLM call is appended to
`partial_calls.jsonl` keyed by a hash of the exact request. Because resume
restores the RNG and memory state of the last checkpoint, the resumed round
re-issues byte-identical requests, which are answered from this cache
instead of the API. The cache is cleared at each checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .parquet import atomic_write_json

CHECKPOINT = "checkpoint.json"
PARTIAL = "partial_calls.jsonl"


def save_checkpoint(run_dir: Path, state: dict) -> None:
    atomic_write_json(run_dir / CHECKPOINT, state)
    partial = run_dir / PARTIAL
    if partial.exists():
        partial.unlink()


def load_checkpoint(run_dir: Path) -> dict | None:
    path = run_dir / CHECKPOINT
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def request_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, run_dir: Path):
        self.path = run_dir / PARTIAL
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line after a crash
                self.entries[rec["key"]] = rec["response"]

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, key: str, response: dict) -> None:
        self.entries[key] = response
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "response": response}, default=str) + "\n")

    def clear(self) -> None:
        self.entries.clear()
        if self.path.exists():
            self.path.unlink()
