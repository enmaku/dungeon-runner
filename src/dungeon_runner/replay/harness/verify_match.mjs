/**
 * Stepwise replay verifier (one envelope per process).
 * Imports portfolio-site web game engine via PORTFOLIO_SITE_ROOT.
 *
 * Usage: node verify_match.mjs <envelope.json>
 * stdout: { "ok": true } | { "ok": false, "failure": { "code", "step?", "detail?" } }
 */
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

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
      const { applyAction, MATCH_PHASES } = kernel
      const { encodeActionIndex } = policy
      const { bootstrapMatchStateForReplay } = replayBootstrap

      let state
      try {
        state = bootstrapMatchStateForReplay(envelope.setup, envelope.seed)
      } catch (err) {
        fail('engine_error', undefined, err.message)
      }

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
        const index = encodeActionIndex(state, action)
        if (index < 0) {
          fail('unmapped_action_type', step)
        }

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

      process.stdout.write(JSON.stringify({ ok: true }) + '\n')
    })
    .catch((err) => {
      fail('engine_error', undefined, err.message)
    })
}

main()
