/**
 * Stepwise replay verifier (one envelope per process).
 * Imports portfolio-site web game engine via PORTFOLIO_SITE_ROOT.
 *
 * Usage: node verify_match.mjs <envelope.json>
 * stdout: { "ok": true } | { "ok": false, "failure": { "code", "step?", "detail?" } }
 */
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { resolve, join } from 'node:path'
import { replayEnvelopeToMatchOver } from './replay_to_match_over.mjs'

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

      const replayed = replayEnvelopeToMatchOver(state, envelope, {
        applyAction,
        encodeActionIndex,
        MATCH_PHASES,
      })
      if (!replayed.ok) {
        const { failure } = replayed
        fail(failure.code, failure.step, failure.detail)
      }

      process.stdout.write(JSON.stringify({ ok: true }) + '\n')
    })
    .catch((err) => {
      fail('engine_error', undefined, err.message)
    })
}

main()
