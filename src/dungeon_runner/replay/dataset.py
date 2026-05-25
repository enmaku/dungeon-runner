"""Dataset build: web-engine labels → derived Parquet per verified match."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from dungeon_runner.replay.eval.eval_suite import (
    EvalSuiteArtifact,
    EvalSuiteError,
    require_eval_suite,
)
from dungeon_runner.replay.eval.split_resolver import split_for
from dungeon_runner.replay.store import raw_path
from dungeon_runner.replay.verify_manifest import load_verify_manifest
from dungeon_runner.replay.web_engine import (
    default_harness_path,
    default_node_command,
    require_portfolio_site_root,
)

DATASET_ENCODING_VERSION = 2

_ROW_SCHEMA = pa.schema(
    [
        ("step", pa.int32()),
        ("seat", pa.string()),
        ("obs", pa.list_(pa.float32())),
        ("mask", pa.list_(pa.int8())),
        ("policy_action_index", pa.int32()),
        ("phase", pa.string()),
        ("subphase", pa.string()),
        ("is_human", pa.bool_()),
        ("model_id", pa.string()),
        ("nn_debug", pa.string()),
        ("match_id", pa.string()),
        ("split", pa.string()),
    ]
)


class DatasetBuildError(RuntimeError):
    """Dataset stage failed; no partial derived artifacts from this run."""


@dataclass(frozen=True)
class DerivedMatchMeta:
    match_id: str
    encoding_version: int
    row_count: int
    built_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "encoding_version": self.encoding_version,
            "row_count": self.row_count,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DerivedMatchMeta:
        return cls(
            match_id=str(data["match_id"]),
            encoding_version=int(data["encoding_version"]),
            row_count=int(data["row_count"]),
            built_at=str(data["built_at"]),
        )


@dataclass
class DatasetSummary:
    built: list[str] = field(default_factory=list)
    retagged: list[str] = field(default_factory=list)


def derived_dir(data_dir: Path, match_id: str) -> Path:
    return data_dir / "derived" / match_id


def derived_meta_path(data_dir: Path, match_id: str) -> Path:
    return derived_dir(data_dir, match_id) / "meta.json"


def derived_rows_path(data_dir: Path, match_id: str) -> Path:
    return derived_dir(data_dir, match_id) / "rows.parquet"


def load_derived_meta(data_dir: Path, match_id: str) -> DerivedMatchMeta | None:
    path = derived_meta_path(data_dir, match_id)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return DerivedMatchMeta.from_dict(data)


def _staging_root(data_dir: Path) -> Path:
    return data_dir / "derived" / ".staging"


def derived_split_tag(data_dir: Path, match_id: str) -> str | None:
    path = derived_rows_path(data_dir, match_id)
    if not path.is_file():
        return None
    table = pq.read_table(path, columns=["split"])
    if table.num_rows == 0:
        return None
    return str(table.column("split")[0].as_py())


def _retag_match_splits(data_dir: Path, match_id: str, split: str) -> None:
    path = derived_rows_path(data_dir, match_id)
    table = pq.read_table(path)
    n = table.num_rows
    idx = table.schema.get_field_index("split")
    if idx < 0:
        raise DatasetBuildError(f"derived rows missing split column: {match_id}")
    table = table.set_column(idx, "split", pa.array([split] * n, type=pa.string()))
    if "match_id" in table.column_names:
        mid_idx = table.schema.get_field_index("match_id")
        table = table.set_column(
            mid_idx, "match_id", pa.array([match_id] * n, type=pa.string())
        )
    pq.write_table(table, path)


def sync_derived_splits(
    data_dir: Path,
    eval_suite: EvalSuiteArtifact,
) -> list[str]:
    """Rewrite split tags in existing derived Parquet to match the eval suite."""
    verify = load_verify_manifest(data_dir)
    retagged: list[str] = []
    for match_id in sorted(verify.verified):
        if load_derived_meta(data_dir, match_id) is None:
            continue
        expected = split_for(match_id, eval_suite)
        current = derived_split_tag(data_dir, match_id)
        if current is None or current == expected:
            continue
        _retag_match_splits(data_dir, match_id, expected)
        retagged.append(match_id)
    return retagged


def pending_dataset_ids(data_dir: Path, *, encode_all: bool = False) -> list[str]:
    verify = load_verify_manifest(data_dir)
    pending: list[str] = []
    for match_id in sorted(verify.verified):
        if not raw_path(data_dir, match_id).is_file():
            continue
        if encode_all:
            pending.append(match_id)
            continue
        meta = load_derived_meta(data_dir, match_id)
        if meta is None or meta.encoding_version < DATASET_ENCODING_VERSION:
            pending.append(match_id)
    return pending


def _parse_harness_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("harness produced empty stdout")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("harness stdout must be a JSON object")
    return payload


def _run_node_harness(
    *,
    envelope_path: Path,
    node_cmd: list[str],
    harness_path: Path,
    portfolio_root: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PORTFOLIO_SITE_ROOT"] = str(portfolio_root)
    proc = subprocess.run(
        [*node_cmd, str(harness_path), str(envelope_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        return {
            "ok": False,
            "failure": {"code": "engine_error", "detail": detail},
        }
    try:
        return _parse_harness_stdout(proc.stdout)
    except (json.JSONDecodeError, ValueError) as err:
        return {
            "ok": False,
            "failure": {
                "code": "engine_error",
                "detail": f"invalid harness JSON: {err}",
            },
        }


def _normalize_harness_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "policy_action_index" not in out and "action" in out:
        out["policy_action_index"] = out.pop("action")
    return out


def _rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    columns: dict[str, list[Any]] = {name: [] for name in _ROW_SCHEMA.names}
    for row in rows:
        normalized = _normalize_harness_row(row)
        for name in _ROW_SCHEMA.names:
            columns[name].append(normalized.get(name))
    return pa.table(columns, schema=_ROW_SCHEMA)


def _tag_rows(
    rows: list[dict[str, Any]],
    *,
    match_id: str,
    eval_suite: EvalSuiteArtifact,
) -> list[dict[str, Any]]:
    split = split_for(match_id, eval_suite)
    tagged: list[dict[str, Any]] = []
    for row in rows:
        tagged.append(
            {
                **row,
                "match_id": match_id,
                "split": split,
            }
        )
    return tagged


def _write_match_artifact(
    staging_dir: Path,
    match_id: str,
    rows: list[dict[str, Any]],
) -> DerivedMatchMeta:
    out_dir = staging_dir / match_id
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _rows_to_table(rows)
    pq.write_table(table, out_dir / "rows.parquet")
    meta = DerivedMatchMeta(
        match_id=match_id,
        encoding_version=DATASET_ENCODING_VERSION,
        row_count=len(rows),
        built_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
    )
    (out_dir / "meta.json").write_text(
        json.dumps(meta.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def _commit_staging(data_dir: Path, match_ids: list[str]) -> None:
    staging = _staging_root(data_dir)
    for match_id in match_ids:
        src = staging / match_id
        dest = derived_dir(data_dir, match_id)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    if staging.exists():
        shutil.rmtree(staging)


def _build_match(
    match_id: str,
    data_dir: Path,
    *,
    node_cmd: list[str],
    harness_path: Path,
    portfolio_root: Path,
    build_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = raw_path(data_dir, match_id)
    runner = build_fn or _run_node_harness
    return runner(
        envelope_path=path,
        node_cmd=node_cmd,
        harness_path=harness_path,
        portfolio_root=portfolio_root,
    )


def run_dataset(
    *,
    data_dir: Path,
    encode_all: bool = False,
    match_ids: list[str] | None = None,
    node_cmd: list[str] | None = None,
    harness_path: Path | None = None,
    portfolio_root: Path | None = None,
    build_fn: Callable[..., dict[str, Any]] | None = None,
) -> DatasetSummary:
    data_dir = data_dir.resolve()
    portfolio_root = portfolio_root or require_portfolio_site_root()
    try:
        eval_suite = require_eval_suite(data_dir)
    except EvalSuiteError as exc:
        raise DatasetBuildError(str(exc)) from exc

    summary = DatasetSummary(retagged=sync_derived_splits(data_dir, eval_suite))

    node_cmd = node_cmd or default_node_command()
    harness_path = harness_path or default_dataset_harness_path()
    if not harness_path.is_file():
        raise RuntimeError(f"dataset harness not found: {harness_path}")

    targets = (
        list(match_ids)
        if match_ids is not None
        else pending_dataset_ids(data_dir, encode_all=encode_all)
    )
    if not targets:
        return summary

    staging = _staging_root(data_dir)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    built_payloads: list[tuple[str, list[dict[str, Any]]]] = []
    try:
        for match_id in targets:
            payload = _build_match(
                match_id,
                data_dir,
                node_cmd=node_cmd,
                harness_path=harness_path,
                portfolio_root=portfolio_root,
                build_fn=build_fn,
            )
            if payload.get("ok") is not True:
                failure = payload.get("failure") or {}
                code = failure.get("code", "engine_error")
                detail = failure.get("detail", "")
                raise DatasetBuildError(
                    f"dataset build failed for {match_id}: {code} {detail}".strip()
                )
            raw_rows = payload.get("rows")
            if not isinstance(raw_rows, list):
                raise DatasetBuildError(
                    f"dataset build failed for {match_id}: harness missing rows"
                )
            rows = _tag_rows(raw_rows, match_id=match_id, eval_suite=eval_suite)
            _write_match_artifact(staging, match_id, rows)
            built_payloads.append((match_id, rows))

        _commit_staging(data_dir, [match_id for match_id, _ in built_payloads])
        summary.built.extend(match_id for match_id, _ in built_payloads)
        return summary
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def default_dataset_harness_path() -> Path:
    return default_harness_path().parent / "build_match_dataset.mjs"
