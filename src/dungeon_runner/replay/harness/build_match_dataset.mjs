/**
 * Dataset label harness (one envelope per process).
 * Replays through web game engine; emits derived training rows on stdout.
 *
 * Usage: node build_match_dataset.mjs <envelope.json>
 * stdout: { ok, encoding_version, human_seat_id, rows } | { ok: false, failure }
 */
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ENCODING_VERSION = 1
const __dirname = dirname(fileURLToPath(import.meta.url))

function fail(code, step = undefined, detail = undefined) {
  const failure = { code }
  if (step !== undefined) failure.step = step
  if (detail !== undefined) failure.detail = detail
  process.stdout.write(JSON.stringify({ ok: false, failure }) + '\n')
  process.exit(0)
}

function featureRoot() {
  const root = process.env.PORTFOLIO_SITE_ROOT
  if (!root) {
    fail('engine_error', undefined, 'PORTFOLIO_SITE_ROOT is not set')
  }
  return resolve(root, 'src/features/dungeon-runner')
}

function subphaseAt(state) {
  if (state.phase === 'bidding') return state.bidding?.subphase ?? null
  if (state.phase === 'dungeon') return state.dungeon?.subphase ?? null
  return null
}

async function loadEngine() {
  const base = featureRoot()
  const kernel = await import(pathToFileURL(join(base, 'engine/kernel.js')).href)
  const policy = await import(pathToFileURL(join(base, 'nn/policyAdapter.js')).href)
  const replayBootstrap = await import(
    pathToFileURL(join(base, 'debug/replayBootstrap.js')).href,
  )
  return { kernel, policy, replayBootstrap }
}

function main() {
  const envelopePath = process.argv[2]
  if (!envelopePath) {
    fail('engine_error', undefined, 'envelope path argument required')
  }

  let envelope
  try {
    envelope = JSON.parse(readFileSync(envelopePath, 'utf8'))
  } catch (err) {
    fail('engine_error', undefined, `failed to read envelope: ${err.message}`)
  }

  loadEngine()
    .then(({ kernel, policy, replayBootstrap }) => {
      const { applyAction, getLegalActions, MATCH_PHASES } = kernel
      const { buildPolicyObservation, buildPolicyLegalMask, encodeActionIndex } = policy
      const { bootstrapMatchStateForReplay } = replayBootstrap

      let state
      try {
        state = bootstrapMatchStateForReplay(envelope.setup, envelope.seed)
      } catch (err) {
        fail('engine_error', undefined, err.message)
      }

      const humanSeat = state.seats.find((seat) => seat.role?.type === 'human')
      const humanSeatId = humanSeat?.id ?? null
      if (!humanSeatId) {
        fail('engine_error', undefined, 'no human seat in initial state')
      }

      const history = envelope.history ?? []
      const rows = []
      let previousAfter = null

      for (let step = 0; step < history.length; step += 1) {
        const entry = history[step]
        const { rngStepBefore, rngStepAfter, actorSeatId, action } = entry
        if (!action || typeof action !== 'object') {
          continue
        }

        if (
          !Number.isInteger(rngStepBefore) ||
          !Number.isInteger(rngStepAfter) ||
          rngStepAfter <= rngStepBefore
        ) {
          fail('rng_chain_break', step)
        }
        if (previousAfter !== null && rngStepBefore !== previousAfter) {
          fail('rng_chain_break', step)
        }
        previousAfter = rngStepAfter

        if (state.rng.step !== rngStepBefore) {
          fail(
            'rng_chain_break',
            step,
            `engine rng step ${state.rng.step} != recorded rngStepBefore ${rngStepBefore}`,
          )
        }

        if (actorSeatId !== state.turn.activeSeatId) {
          fail('actor_mismatch', step)
        }

        const actor = { seatId: actorSeatId }
        const legalActions = getLegalActions(state, actor)
        const obs = buildPolicyObservation(state, actor)
        const mask = buildPolicyLegalMask(state, actor, legalActions)
        const actionIndex = encodeActionIndex(state, action)
        if (actionIndex < 0) {
          fail('unmapped_action_type', step)
        }

        const isHuman = actorSeatId === humanSeatId
        const modelId =
          typeof action.modelId === 'string' && action.modelId.length > 0
            ? action.modelId
            : null

        rows.push({
          step,
          seat: actorSeatId,
          obs,
          mask,
          policy_action_index: actionIndex,
          phase: state.phase,
          subphase: subphaseAt(state),
          is_human: isHuman,
          model_id: modelId,
          nn_debug: null,
        })

        const result = applyAction(state, action, actor)
        if (!result.ok) {
          fail('illegal_action', step, result.errorCode ?? 'applyAction rejected')
        }

        if (result.state.rng.step !== rngStepAfter) {
          fail(
            'rng_chain_break',
            step,
            `engine rng step ${result.state.rng.step} != recorded rngStepAfter ${rngStepAfter}`,
          )
        }

        state = result.state
      }

      if (state.phase !== MATCH_PHASES.MATCH_OVER) {
        fail('match_not_over')
      }

      process.stdout.write(
        JSON.stringify({
          ok: true,
          encoding_version: ENCODING_VERSION,
          human_seat_id: humanSeatId,
          rows,
        }) + '\n',
      )
    })
    .catch((err) => {
      fail('engine_error', undefined, err.message)
    })
}

main()
