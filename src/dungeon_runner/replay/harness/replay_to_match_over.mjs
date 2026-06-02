/**
 * Stepwise replay through web game engine until match-over or failure.
 */
import { resolveReplayStepApplySeat } from './replay_step_apply_seat.mjs'

/**
 * @param {import('../engine/kernel.js').MatchState} state
 * @param {object} envelope
 * @param {{
 *   applyAction: Function,
 *   encodeActionIndex: Function,
 *   MATCH_PHASES: { MATCH_OVER: string },
 * }} engine
 * @returns {{ ok: true, state: object } | { ok: false, failure: { code: string, step?: number, detail?: string } }}
 */
export function replayEnvelopeToMatchOver(state, envelope, { applyAction, encodeActionIndex, MATCH_PHASES }) {
  const history = envelope.history ?? []
  let previousAfter = null

  for (let step = 0; step < history.length; step += 1) {
    const entry = history[step]
    const { rngStepBefore, rngStepAfter, actorSeatId, action } = entry

    if (
      !Number.isInteger(rngStepBefore) ||
      !Number.isInteger(rngStepAfter) ||
      rngStepAfter <= rngStepBefore
    ) {
      return { ok: false, failure: { code: 'rng_chain_break', step } }
    }
    if (previousAfter !== null && rngStepBefore !== previousAfter) {
      return { ok: false, failure: { code: 'rng_chain_break', step } }
    }
    previousAfter = rngStepAfter

    if (state.rng.step !== rngStepBefore) {
      return {
        ok: false,
        failure: {
          code: 'rng_chain_break',
          step,
          detail: `engine rng step ${state.rng.step} != recorded rngStepBefore ${rngStepBefore}`,
        },
      }
    }

    let applySeatId
    try {
      const resolved = resolveReplayStepApplySeat(state, { actorSeatId, action })
      applySeatId = resolved.applySeatId
      if (!resolved.skipActorMismatchCheck && actorSeatId !== state.turn.activeSeatId) {
        return { ok: false, failure: { code: 'actor_mismatch', step } }
      }
    } catch (err) {
      return { ok: false, failure: { code: 'engine_error', step, detail: err.message } }
    }

    const actor = { seatId: applySeatId }
    const index = encodeActionIndex(state, action)
    if (index < 0) {
      return { ok: false, failure: { code: 'unmapped_action_type', step } }
    }

    const result = applyAction(state, action, actor)
    if (!result.ok) {
      return {
        ok: false,
        failure: { code: 'illegal_action', step, detail: result.errorCode ?? 'applyAction rejected' },
      }
    }

    if (result.state.rng.step !== rngStepAfter) {
      return {
        ok: false,
        failure: {
          code: 'rng_chain_break',
          step,
          detail: `engine rng step ${result.state.rng.step} != recorded rngStepAfter ${rngStepAfter}`,
        },
      }
    }

    state = result.state
  }

  if (state.phase !== MATCH_PHASES.MATCH_OVER) {
    return { ok: false, failure: { code: 'match_not_over' } }
  }

  return { ok: true, state }
}
