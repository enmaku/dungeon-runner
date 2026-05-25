import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { resolveReplayStepApplySeat } from '../../../src/dungeon_runner/replay/harness/replay_step_apply_seat.mjs'

describe('resolveReplayStepApplySeat', () => {
  it('pick-adventurer CHOOSE_NEXT_ADVENTURER uses picker seat and skips checks', () => {
    const state = {
      phase: 'pick-adventurer',
      pickAdventurer: { activeSeatId: 'seat-picker' },
      turn: { activeSeatId: 'seat-turn' },
    }
    const result = resolveReplayStepApplySeat(state, {
      actorSeatId: 'seat-recorded',
      action: { type: 'CHOOSE_NEXT_ADVENTURER', hero: 'WARRIOR' },
    })
    assert.equal(result.skipActorMismatchCheck, true)
    assert.equal(result.applySeatId, 'seat-picker')
    assert.equal(result.skipDatasetRow, true)
  })

  it('bidding PASS uses recorded actor and does not skip', () => {
    const state = {
      phase: 'bidding',
      turn: { activeSeatId: 'seat-1' },
    }
    const result = resolveReplayStepApplySeat(state, {
      actorSeatId: 'seat-1',
      action: { type: 'PASS' },
    })
    assert.equal(result.skipActorMismatchCheck, false)
    assert.equal(result.applySeatId, 'seat-1')
    assert.equal(result.skipDatasetRow, false)
  })

  it('pick-adventurer CHOOSE_NEXT_ADVENTURER throws when picker seat missing', () => {
    const state = {
      phase: 'pick-adventurer',
      pickAdventurer: { activeSeatId: null },
      turn: { activeSeatId: 'seat-1' },
    }
    assert.throws(
      () =>
        resolveReplayStepApplySeat(state, {
          actorSeatId: 'seat-1',
          action: { type: 'CHOOSE_NEXT_ADVENTURER', hero: 'WARRIOR' },
        }),
      (err) => err instanceof Error && err.message.includes('missing picker seat'),
    )
  })
})
