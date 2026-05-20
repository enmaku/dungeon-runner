"""Derived store Parquet loader errors."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dungeon_runner.replay.eval.derived_store import (
    DerivedStoreError,
    load_match_rows,
)


def test_load_match_rows_missing_file(tmp_path: Path):
    with pytest.raises(DerivedStoreError, match="derived rows missing"):
        load_match_rows(tmp_path / "rows.parquet")


def test_load_match_rows_missing_columns(tmp_path: Path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"split": ["val"]}), path)
    with pytest.raises(DerivedStoreError, match="missing columns"):
        load_match_rows(path)
