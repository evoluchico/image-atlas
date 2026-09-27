"""Atomic Parquet/JSON writing helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, default=str).encode("utf-8"))


def write_records(path: Path, records: list[dict], schema: pa.Schema) -> None:
    cols = {f.name: [r.get(f.name) for r in records] for f in schema}
    table = pa.Table.from_pydict(cols, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def write_frame(path: Path, df: pd.DataFrame, csv: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    if csv:
        df.to_csv(path.with_suffix(".csv"), index=False)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
