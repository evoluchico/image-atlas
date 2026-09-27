"""Event logging: tidy tables plus raw API records for every run."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from ..storage import schemas
from ..storage.parquet import write_records


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLogger:
    """Accumulates the run's tidy records and flushes them at each checkpoint.

    Raw API records (rendered system prompt, rendered context, request body,
    raw response) are buffered per round and written to raw/round_XXX.jsonl.gz.
    """

    TABLES = ("messages", "exposures", "generation_events", "probes")

    def __init__(self, run_id: str, run_dir: Path, save_raw: bool = True):
        self.run_id = run_id
        self.run_dir = run_dir
        self.save_raw = save_raw
        self.records: dict[str, list[dict]] = {t: [] for t in self.TABLES}
        self.raw: list[dict] = []

    def add(self, table: str, rows: list[dict]) -> None:
        self.records[table].extend(rows)

    def add_raw(self, rec: dict) -> None:
        if self.save_raw:
            self.raw.append(rec)

    def flush(self, round_id: int) -> None:
        for table in self.TABLES:
            write_records(self.run_dir / f"{table}.parquet", self.records[table], schemas.TABLES[table])
        if self.save_raw and self.raw:
            raw_dir = self.run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            rows = sorted(self.raw, key=lambda r: (r["phase"], r["agent_id"], r.get("attempt", 0)))
            with gzip.open(raw_dir / f"round_{round_id:03d}.jsonl.gz", "wt", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
        self.raw = []

    def restore(self, max_round: int) -> None:
        """Reload tables written at the last checkpoint, dropping any rows from
        a round that was in progress (tables are written before the checkpoint
        file, so they can briefly run ahead of it)."""
        for table in self.TABLES:
            path = self.run_dir / f"{table}.parquet"
            if not path.exists():
                continue
            rows = pq.read_table(path).to_pylist()
            self.records[table] = [r for r in rows if r["round"] <= max_round]
