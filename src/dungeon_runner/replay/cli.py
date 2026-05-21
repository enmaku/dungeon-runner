"""Replay pipeline CLI: `python -m dungeon_runner.replay.cli <stage>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dungeon_runner.replay.env import load_dotenv, require_database_url
from dungeon_runner.replay.eval.eval_config import EvalConfigError, init_eval_config
from dungeon_runner.replay.eval.eval_suite import (
    DEFAULT_SAMPLING_SEED,
    EvalSuiteError,
    init_eval_suite,
)
from dungeon_runner.replay.bc import BCPrerequisiteError, BCStageError, default_bc_run_id, run_bc
from dungeon_runner.replay.ppo import (
    PPOPrerequisiteError,
    PPOStageError,
    default_ppo_run_id,
    run_ppo,
)
from dungeon_runner.replay.publish import PublishError, run_publish
from dungeon_runner.replay.dataset import DatasetBuildError, run_dataset
from dungeon_runner.replay.ingest import run_ingest
from dungeon_runner.replay import progress
from dungeon_runner.replay.run_all import run_all
from dungeon_runner.replay.verify import run_verify

DEFAULT_DATA_DIR = Path("data/replays")

_NOT_IMPLEMENTED_STAGES: tuple[str, ...] = ()

_STAGE_STUB_NOTES: dict[str, str] = {}


def _cmd_ingest(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    from_export = Path(args.from_export) if args.from_export else None
    try:
        database_url = None if from_export else require_database_url()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    summary = run_ingest(
        data_dir=data_dir,
        from_export=from_export,
        database_url=database_url,
    )
    if summary.ingested:
        print(f"ingested {len(summary.ingested)}: {', '.join(summary.ingested)}")
    if summary.skipped:
        print(f"skipped {len(summary.skipped)}")
        for entry in summary.skipped:
            print(f"  {entry['id']}: {entry['reason']}")
    if not summary.ingested and not summary.skipped:
        print("no new matches")
    return 0


def _cmd_eval_suite_init(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    try:
        artifact = init_eval_suite(data_dir, sampling_seed=args.sampling_seed)
    except EvalSuiteError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"eval suite v{artifact.suite_version}: "
        f"{len(artifact.val_match_ids)} val / "
        f"{len(artifact.created_from_match_ids)} verified "
        f"(seed={artifact.sampling_seed})"
    )
    for match_id in artifact.val_match_ids:
        print(f"  val: {match_id}")
    return 0


def _cmd_eval_config_init(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    try:
        artifact = init_eval_config(data_dir, overwrite=args.overwrite)
    except EvalConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"eval config: {len(artifact.sim_seeds)} sim seeds, "
        f"ε={artifact.sim_regression_tolerance}, "
        f"floor={artifact.replay_accuracy_floor!r}"
    )
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    try:
        summary = run_dataset(data_dir=data_dir, encode_all=args.all)
    except DatasetBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if summary.retagged:
        print(
            f"retagged splits for {len(summary.retagged)}: "
            f"{', '.join(summary.retagged)}"
        )
    if summary.built:
        print(f"built {len(summary.built)}: {', '.join(summary.built)}")
    elif not summary.retagged:
        print("no pending dataset matches")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    if not args.run:
        print("publish requires --run", file=sys.stderr)
        return 1
    run_dir = Path(args.run)
    data_dir = Path(args.data_dir)
    try:
        summary = run_publish(
            run_dir=run_dir,
            data_dir=data_dir,
            version_override=args.version,
        )
    except PublishError as exc:
        if exc.reasons:
            print(f"publish failed: {', '.join(exc.reasons)}", file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        return 1
    print(f"promoted {summary.run_id} → {summary.promoted_version}")
    print(f"  {summary.version_dir}")
    return 0


def _cmd_bc(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    run_id = args.run_id or default_bc_run_id()
    try:
        summary = run_bc(
            data_dir=data_dir,
            run_id=run_id,
            gate_preview=not args.no_gate_preview,
        )
    except BCPrerequisiteError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except BCStageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            f"BC training requires TensorFlow: pip install -e \".[train]\" ({exc})",
            file=sys.stderr,
        )
        return 1
    print(f"bc run {summary.run_id}: {summary.run_dir}")
    tb_dir = summary.run_dir / "tb"
    if tb_dir.is_dir():
        progress.log_tensorboard(tb_dir, run_label=summary.run_id)
    if summary.floor_outcome:
        print(f"floor recorder: {summary.floor_outcome}")
    if summary.gate_preview_passed is not None:
        status = "pass" if summary.gate_preview_passed else "fail"
        reasons = ", ".join(summary.gate_preview_reasons) or "ok"
        print(f"gate preview: {status} ({reasons})")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    try:
        summary = run_verify(data_dir=data_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if summary.verified:
        print(f"verified {len(summary.verified)}: {', '.join(summary.verified)}")
    if summary.failed:
        print(f"failed {len(summary.failed)}")
        for entry in summary.failed:
            reason = entry["reason"]
            code = reason.get("code", "?")
            step = reason.get("step")
            suffix = f" step={step}" if step is not None else ""
            print(f"  {entry['id']}: {code}{suffix}")
    if not summary.verified and not summary.failed:
        print("no pending verify matches")
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    return run_all(
        data_dir=Path(args.data_dir),
        with_ppo=args.with_ppo,
        with_publish=args.with_publish,
    )


def _cmd_ppo(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    bc_run = Path(args.bc_run)
    run_id = args.run_id or default_ppo_run_id()
    try:
        summary = run_ppo(
            data_dir=data_dir,
            bc_run=bc_run,
            run_id=run_id,
            bc_anchor_lambda=args.bc_anchor_lambda,
            bc_anchor_beta=args.bc_anchor_beta,
            max_updates=args.max_updates,
            use_ray=not args.no_ray,
            ray_workers=args.ray_workers,
            gate_preview=not args.no_gate_preview,
        )
    except PPOPrerequisiteError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except PPOStageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            f"PPO training requires TensorFlow: pip install -e \".[train]\" ({exc})",
            file=sys.stderr,
        )
        return 1
    print(f"ppo run {summary.run_id}: {summary.run_dir}")
    tb_dir = summary.run_dir / "tb"
    if tb_dir.is_dir():
        progress.log_tensorboard(tb_dir, run_label=summary.run_id)
    reg = "pass" if summary.regression_passed else "fail"
    print(f"ppo bc regression: {reg}")
    if summary.gate_preview_passed is not None:
        status = "pass" if summary.gate_preview_passed else "fail"
        reasons = ", ".join(summary.gate_preview_reasons) or "ok"
        print(f"gate preview: {status} ({reasons})")
    return 0 if summary.regression_passed else 1


def _cmd_not_implemented(stage: str) -> int:
    note = _STAGE_STUB_NOTES.get(stage, "")
    msg = f"stage {stage!r} is not implemented yet"
    if note:
        msg += f"; when shipped: {note}"
    msg += "; see docs/replay-pipeline.md"
    print(msg, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="dungeon_runner.replay.cli")
    sub = parser.add_subparsers(dest="stage", required=True)

    ingest = sub.add_parser("ingest", help="Pull completed match replays into raw store")
    ingest.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    ingest.add_argument(
        "--from-export",
        metavar="PATH",
        help="Offline ingest from Firebase export JSON (top-level match-id map)",
    )
    ingest.set_defaults(handler=_cmd_ingest)

    verify = sub.add_parser("verify", help="Replay ingested envelopes through web game engine")
    verify.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    verify.set_defaults(handler=_cmd_verify)

    dataset = sub.add_parser(
        "dataset",
        help="Build derived Parquet rows from verified replays (web engine labels)",
    )
    dataset.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    dataset.add_argument(
        "--all",
        action="store_true",
        help="Re-encode every verified match (default: pending only)",
    )
    dataset.set_defaults(handler=_cmd_dataset)

    eval_suite = sub.add_parser(
        "eval_suite",
        help="Frozen holdout suite from verify manifest verified ids",
    )
    eval_suite_sub = eval_suite.add_subparsers(dest="eval_suite_cmd", required=True)
    eval_suite_init = eval_suite_sub.add_parser(
        "init",
        help="Sample ~20% of verified ids into eval_suite.json",
    )
    eval_suite_init.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    eval_suite_init.add_argument(
        "--sampling-seed",
        type=int,
        default=DEFAULT_SAMPLING_SEED,
        help=f"Seeded holdout sampling (default: {DEFAULT_SAMPLING_SEED})",
    )
    eval_suite_init.set_defaults(handler=_cmd_eval_suite_init)

    eval_config = sub.add_parser(
        "eval_config",
        help="Sim seeds, regression tolerance, replay accuracy floor",
    )
    eval_config_sub = eval_config.add_subparsers(dest="eval_config_cmd", required=True)
    eval_config_init = eval_config_sub.add_parser(
        "init",
        help="Write eval_config.json with default seeds and ε",
    )
    eval_config_init.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    eval_config_init.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing eval_config.json",
    )
    eval_config_init.set_defaults(handler=_cmd_eval_config_init)

    bc = sub.add_parser("bc", help="BC policy training → training run artifact")
    bc.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    bc.add_argument(
        "--run-id",
        help="Training run id (default: bc-<UTC compact>)",
    )
    bc.add_argument(
        "--no-gate-preview",
        action="store_true",
        help="Skip gate evaluator preview",
    )
    bc.set_defaults(handler=_cmd_bc)

    ppo = sub.add_parser("ppo", help="BC-anchored PPO training → training run artifact")
    ppo.add_argument(
        "--bc-run",
        required=True,
        help="BC training run artifact directory (init weights + regression baseline)",
    )
    ppo.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    ppo.add_argument(
        "--run-id",
        help="Training run id (default: ppo-<UTC compact>)",
    )
    ppo.add_argument(
        "--bc-anchor-lambda",
        type=float,
        default=0.1,
        help="BC anchor CE strength (0 skips derived-store prerequisites)",
    )
    ppo.add_argument(
        "--bc-anchor-beta",
        type=float,
        default=0.0,
        help="BC anchor KL vs frozen teacher (0 disables)",
    )
    ppo.add_argument(
        "--max-updates",
        type=int,
        default=16,
        help="PPO rollout/update steps (best val checkpoint restored at end)",
    )
    ppo.add_argument(
        "--ray-workers",
        type=int,
        default=8,
        help="Parallel PPO rollout workers when Ray is enabled",
    )
    ppo.add_argument(
        "--no-ray",
        action="store_true",
        help="Single-process rollout collection",
    )
    ppo.add_argument(
        "--no-gate-preview",
        action="store_true",
        help="Skip gate evaluator preview",
    )
    ppo.set_defaults(handler=_cmd_ppo)

    publish = sub.add_parser(
        "publish",
        help="Gated promotion from committed training run artifact",
    )
    publish.add_argument(
        "--run",
        required=True,
        help="Committed training run artifact directory (not *.tmp)",
    )
    publish.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root for eval config (default: {DEFAULT_DATA_DIR})",
    )
    publish.add_argument(
        "--version",
        help="Manual promoted version (e.g. v0.3); default auto-bump",
    )
    publish.set_defaults(handler=_cmd_publish)

    run_all_parser = sub.add_parser(
        "run-all",
        help="Orchestrate ingest → verify → eval_* → dataset → bc [→ ppo] [→ publish]",
    )
    run_all_parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Training data root (default: {DEFAULT_DATA_DIR})",
    )
    run_all_parser.add_argument(
        "--with-ppo",
        action="store_true",
        help="Chain BC-anchored PPO after bc",
    )
    run_all_parser.add_argument(
        "--with-publish",
        action="store_true",
        help="Chain gated promotion on last train artifact",
    )
    run_all_parser.set_defaults(handler=_cmd_run_all)

    for stage in _NOT_IMPLEMENTED_STAGES:
        stub_help = _STAGE_STUB_NOTES.get(stage, "future pipeline stage")
        stub = sub.add_parser(stage, help=f"(not implemented) {stub_help}")
        stub.add_argument(
            "--data-dir",
            default=str(DEFAULT_DATA_DIR),
            help=f"Training data root (default: {DEFAULT_DATA_DIR})",
        )
        stub.set_defaults(handler=lambda _a, s=stage: _cmd_not_implemented(s))

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
