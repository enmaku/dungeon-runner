"""Load derived training rows from per-match Parquet (dataset build output)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from dungeon_runner.replay.eval.replay_metrics import DerivedRow


class DerivedStoreError(ValueError):
    """Derived store read failed."""


@dataclass(frozen=True)
class ParquetDerivedRow:
    match_id: str
    split: str
    is_human_step: bool
    obs: Any
    mask: Any
    policy_action_index: int


def _row_value(column: Any, index: int) -> Any:
    value = column[index]
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def _table_to_rows(table: Any, match_id: str) -> list[ParquetDerivedRow]:
    cols = {name: table.column(name) for name in table.column_names}
    required = ("split", "is_human", "policy_action_index", "obs", "mask")
    missing = [name for name in required if name not in cols]
    if missing:
        raise DerivedStoreError(f"rows.parquet missing columns: {', '.join(missing)}")

    rows: list[ParquetDerivedRow] = []
    for i in range(table.num_rows):
        mid = match_id
        if "match_id" in cols:
            mid = str(_row_value(cols["match_id"], i))
        rows.append(
            ParquetDerivedRow(
                match_id=mid,
                split=str(_row_value(cols["split"], i)),
                is_human_step=bool(_row_value(cols["is_human"], i)),
                obs=_row_value(cols["obs"], i),
                mask=_row_value(cols["mask"], i),
                policy_action_index=int(_row_value(cols["policy_action_index"], i)),
            )
        )
    return rows


def load_match_rows(parquet_path: Path) -> list[ParquetDerivedRow]:
    if not parquet_path.is_file():
        raise DerivedStoreError(f"derived rows missing: {parquet_path}")
    table = pq.read_table(parquet_path)
    return _table_to_rows(table, match_id=parquet_path.parent.name)


def load_derived_rows(
    data_dir: Path,
    *,
    match_ids: set[str] | list[str] | None = None,
) -> Iterator[DerivedRow]:
    derived_root = data_dir / "derived"
    if not derived_root.is_dir():
        return

    ids_filter: set[str] | None
    if match_ids is None:
        ids_filter = None
    else:
        ids_filter = set(match_ids)

    for match_dir in sorted(derived_root.iterdir()):
        if not match_dir.is_dir():
            continue
        match_id = match_dir.name
        if ids_filter is not None and match_id not in ids_filter:
            continue
        parquet_path = match_dir / "rows.parquet"
        if not parquet_path.is_file():
            continue
        yield from load_match_rows(parquet_path)
