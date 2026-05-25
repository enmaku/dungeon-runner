/**
 * Resolves which seat applies a replay history step and whether harness checks
 * should treat pick-adventurer CHOOSE_NEXT_ADVENTURER rows specially.
 */
export function resolveReplayStepApplySeat(state, { actorSeatId, action }) {
  const isPickChoose =
    state.phase === 'pick-adventurer' && action?.type === 'CHOOSE_NEXT_ADVENTURER'

  if (isPickChoose) {
    const applySeatId = state.pickAdventurer?.activeSeatId
    if (typeof applySeatId !== 'string') {
      throw new Error('missing picker seat')
    }
    return {
      applySeatId,
      skipActorMismatchCheck: true,
      skipDatasetRow: true,
    }
  }

  return {
    applySeatId: actorSeatId,
    skipActorMismatchCheck: false,
    skipDatasetRow: false,
  }
}
