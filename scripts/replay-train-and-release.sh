#!/usr/bin/env bash
# Full replay training pipeline: ingest → … → bc → ppo → publish → portfolio-site TF.js sync.
# Exits 0 only when PPO BC regression passes, gated promotion succeeds, and sync completes.
#
# Usage:
#   ./scripts/replay-train-and-release.sh
#   CLEAN_TRAINING=1 ./scripts/replay-train-and-release.sh
#
# Requires .env at repo root (see .env.example):
#   FIREBASE_DATABASE_URL, PORTFOLIO_SITE_ROOT
#
# Optional env:
#   DATA_DIR              default: data/replays
#   BC_ANCHOR_LAMBDA      default: 1.0
#   BC_ANCHOR_BETA        default: 0.1
#   PPO_MAX_UPDATES       default: 16
#   RAY_WORKERS           default: 8
#   PPO_NO_RAY            set to 1 for single-process rollouts
#   CLEAN_TRAINING        set to 1 to wipe models/runs/* and eval_config.json first
#   SKIP_PORTFOLIO_SYNC   set to 1 to skip npm sync (still runs publish)
#   PYTHON                override python binary

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-data/replays}"
BC_ANCHOR_LAMBDA="${BC_ANCHOR_LAMBDA:-1.0}"
BC_ANCHOR_BETA="${BC_ANCHOR_BETA:-0.1}"
PPO_MAX_UPDATES="${PPO_MAX_UPDATES:-16}"
RAY_WORKERS="${RAY_WORKERS:-8}"
CLEAN_TRAINING="${CLEAN_TRAINING:-0}"
SKIP_PORTFOLIO_SYNC="${SKIP_PORTFOLIO_SYNC:-0}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

replay_cli() {
  if [[ $# -lt 1 ]]; then
    die "replay_cli: missing stage"
  fi
  local stage="$1"
  shift
  "$PYTHON" -m dungeon_runner.replay.cli "$stage" --data-dir "$DATA_DIR" "$@"
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name is not set (add to .env or export)"
}

require_env FIREBASE_DATABASE_URL
require_env PORTFOLIO_SITE_ROOT
[[ -d "$PORTFOLIO_SITE_ROOT" ]] || die "PORTFOLIO_SITE_ROOT is not a directory: $PORTFOLIO_SITE_ROOT"

if [[ "$CLEAN_TRAINING" == "1" ]]; then
  log "CLEAN_TRAINING=1: removing models/runs/* and $DATA_DIR/eval_config.json"
  rm -rf "$REPO_ROOT/models/runs/"*
  rm -f "$REPO_ROOT/$DATA_DIR/eval_config.json"
fi

log "repo: $REPO_ROOT"
log "python: $PYTHON"
log "data-dir: $DATA_DIR"
log "ppo: updates=$PPO_MAX_UPDATES anchor lambda=$BC_ANCHOR_LAMBDA beta=$BC_ANCHOR_BETA"

log "stage 1/4: run-all (ingest → verify → eval → dataset → bc)"
replay_cli run-all

BC_RUN="$(find "$REPO_ROOT/models/runs" -maxdepth 1 -type d -name 'bc-*' -print 2>/dev/null | sort -r | head -n 1)"
[[ -n "$BC_RUN" && -f "$BC_RUN/policy.weights.h5" ]] || die "no bc training run artifact under models/runs/"
log "bc artifact: $BC_RUN"

PPO_ARGS=(
  --bc-run "$BC_RUN"
  --bc-anchor-lambda "$BC_ANCHOR_LAMBDA"
  --bc-anchor-beta "$BC_ANCHOR_BETA"
  --max-updates "$PPO_MAX_UPDATES"
  --ray-workers "$RAY_WORKERS"
)
if [[ "${PPO_NO_RAY:-0}" == "1" ]]; then
  PPO_ARGS+=(--no-ray)
fi

log "stage 2/4: ppo (BC-anchored fine-tuning)"
set +e
replay_cli ppo "${PPO_ARGS[@]}"
PPO_EXIT=$?
set -e

PPO_RUN="$(find "$REPO_ROOT/models/runs" -maxdepth 1 -type d -name 'ppo-*' -print 2>/dev/null | sort -r | head -n 1)"
[[ -n "$PPO_RUN" && -f "$PPO_RUN/metrics.json" ]] || die "no ppo training run artifact under models/runs/"
log "ppo artifact: $PPO_RUN"

if ! "$PYTHON" -c "
import json, sys
m = json.loads(open('$PPO_RUN/metrics.json').read())
sys.exit(0 if (m.get('ppo_bc_regression') or {}).get('pass') is True else 1)
"; then
  die "PPO BC regression check failed (metrics: $PPO_RUN/metrics.json). Not publishing."
fi
if [[ "$PPO_EXIT" -ne 0 ]]; then
  die "ppo CLI exited $PPO_EXIT despite regression pass flag; inspect $PPO_RUN"
fi

log "stage 3/4: publish (gated promotion)"
PUBLISH_OUT="$(replay_cli publish --run "$PPO_RUN")"
printf '%s\n' "$PUBLISH_OUT"
PROMOTED_VERSION="$(printf '%s\n' "$PUBLISH_OUT" | sed -n 's/^promoted .* → \(.*\)$/\1/p')"
[[ -n "$PROMOTED_VERSION" ]] || die "could not parse promoted version from publish output"

if [[ "$SKIP_PORTFOLIO_SYNC" == "1" ]]; then
  log "SKIP_PORTFOLIO_SYNC=1: done after publish ($PROMOTED_VERSION)"
  exit 0
fi

log "stage 4/4: portfolio-site TF.js sync ($PROMOTED_VERSION)"
export DUNGEON_RUNNER_ROOT="$REPO_ROOT"
export PYTHON_BIN="$PYTHON"
(
  cd "$PORTFOLIO_SITE_ROOT"
  npm run sync-dungeon-runner-model -- --from-latest
)

log "complete: promoted $PROMOTED_VERSION and synced to portfolio-site (web deployed latest)"
