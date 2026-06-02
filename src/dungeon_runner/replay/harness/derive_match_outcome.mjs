/**
 * Replay envelope → terminal web engine state → completed match outcome JSON.
 * Imports portfolio-site via PORTFOLIO_SITE_ROOT.
 *
 * Usage:
 *   node derive_match_outcome.mjs <envelope.json|-)> [matchId]
 * stdout: single completed match outcome object
 * stderr: { "failure": { "code", "step?", "detail?" } } on error (exit 1)
 */
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { resolve, join } from 'node:path'
import { replayEnvelopeToMatchOver } from './replay_to_match_over.mjs'

function fail(code, step = undefined, detail = undefined) {
  const failure = { code }
  if (step !== undefined) failure.step = step
  if (detail !== undefined) failure.detail = detail
  process.stderr.write(JSON.stringify({ failure }) + '\n')
  process.exit(1)
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
  const analytics = await import(
    pathToFileURL(join(base, 'analytics/buildMatchOutcomeRecord.js')).href,
  )
  return { kernel, policy, replayBootstrap, analytics }
}

function readEnvelopeFromArgv() {
  const source = process.argv[2]
  if (!source) {
    fail('engine_error', undefined, 'envelope path argument required (file path or - for stdin)')
  }
  try {
    const raw =
      source === '-'
        ? readFileSync(0, 'utf8')
        : readFileSync(source, 'utf8')
    return JSON.parse(raw)
  } catch (err) {
    fail('engine_error', undefined, `failed to read envelope: ${err.message}`)
  }
}

function resolveMatchId(envelope, envelopePath) {
  const fromArgv = process.argv[3]
  if (typeof fromArgv === 'string' && fromArgv.length > 0) {
    return fromArgv
  }
  if (typeof envelope.matchId === 'string' && envelope.matchId.length > 0) {
    return envelope.matchId
  }
  if (envelopePath && envelopePath !== '-') {
    const base = envelopePath.replace(/\\/g, '/').split('/').pop() ?? ''
    const stem = base.replace(/\.json$/i, '')
    if (stem) return stem
  }
  fail('engine_error', undefined, 'matchId required (argv[3], envelope.matchId, or envelope filename)')
}

function resolveHumanPlayerSeatId(seats) {
  return seats?.find((seat) => seat.role?.type === 'human')?.id ?? ''
}

function main() {
  const envelopePath = process.argv[2]
  const envelope = readEnvelopeFromArgv()
  const matchId = resolveMatchId(envelope, envelopePath)

  loadEngine()
    .then(({ kernel, policy, replayBootstrap, analytics }) => {
      const { applyAction, MATCH_PHASES } = kernel
      const { encodeActionIndex } = policy
      const { bootstrapMatchStateForReplay } = replayBootstrap
      const { buildMatchOutcomeRecord } = analytics

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

      state = replayed.state
      const seats = state.seats ?? []
      const humanPlayerSeatId = resolveHumanPlayerSeatId(seats)
      if (!humanPlayerSeatId) {
        fail('engine_error', undefined, 'no human seat in terminal state')
      }

      const createdAt =
        typeof envelope.createdAt === 'string' && envelope.createdAt.length > 0
          ? envelope.createdAt
          : undefined

      const record = buildMatchOutcomeRecord({
        matchId,
        createdAt,
        setup: envelope.setup,
        state,
        seats,
        humanPlayerSeatId,
        presentationSpeedProfile: envelope.presentationSpeedProfile,
      })

      process.stdout.write(JSON.stringify(record) + '\n')
    })
    .catch((err) => {
      fail('engine_error', undefined, err.message)
    })
}

main()
