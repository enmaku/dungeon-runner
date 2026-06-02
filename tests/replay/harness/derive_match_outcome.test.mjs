import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, it } from 'node:test'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = join(__dirname, '../../..')
const HARNESS = join(
  REPO_ROOT,
  'src/dungeon_runner/replay/harness/derive_match_outcome.mjs',
)
const FIXTURES = join(REPO_ROOT, 'tests/fixtures/replay')
const OUTCOME_PARITY = join(FIXTURES, 'outcome-parity')

const PORTFOLIO_ROOT = process.env.PORTFOLIO_SITE_ROOT?.trim() ?? ''
const HAS_PORTFOLIO =
  PORTFOLIO_ROOT.length > 0 &&
  (() => {
    try {
      readFileSync(
        join(
          PORTFOLIO_ROOT,
          'src/features/dungeon-runner/analytics/buildMatchOutcomeRecord.js',
        ),
      )
      return true
    } catch {
      return false
    }
  })()

const PARITY_VARIANTS = [
  'victory',
  'defeat-not-eliminated',
  'elimination-end-human',
]

const STRUCTURED_FAILURE_FIXTURES = {
  'match-not-over.json': 'match_not_over',
  'rng-chain-break.json': 'rng_chain_break',
  'actor-mismatch.json': 'actor_mismatch',
  'illegal-action.json': 'illegal_action',
  'unmapped-action-type.json': 'unmapped_action_type',
}

function runHarness(args, env = process.env) {
  return spawnSync(process.execPath, [HARNESS, ...args], {
    encoding: 'utf8',
    env,
    cwd: REPO_ROOT,
  })
}

function portfolioFixturesDir() {
  return join(PORTFOLIO_ROOT, 'src/features/dungeon-runner/analytics/fixtures')
}

function resolveReplayEnvelope(variant) {
  const portfolioPath = join(portfolioFixturesDir(), `replay-envelope-outcome-${variant}.json`)
  try {
    readFileSync(portfolioPath)
    return portfolioPath
  } catch {
    return join(OUTCOME_PARITY, `replay-envelope-outcome-${variant}.json`)
  }
}

function resolveOutcomeGolden(variant) {
  return join(portfolioFixturesDir(), `outcome-${variant}.json`)
}

describe('derive_match_outcome.mjs', () => {
  it('fails fast when PORTFOLIO_SITE_ROOT is unset', () => {
    const env = { ...process.env }
    delete env.PORTFOLIO_SITE_ROOT
    const proc = runHarness(
      [join(FIXTURES, 'valid-match-over-seed42.json')],
      env,
    )
    assert.notEqual(proc.status, 0)
    const err = JSON.parse(proc.stderr.trim())
    assert.equal(err.failure.code, 'engine_error')
    assert.match(err.failure.detail ?? '', /PORTFOLIO_SITE_ROOT/)
    assert.equal(proc.stdout.trim(), '')
  })

  it('reports match_not_over with non-zero exit', () => {
    if (!HAS_PORTFOLIO) return
    const proc = runHarness(
      [join(FIXTURES, 'match-not-over.json')],
      { ...process.env, PORTFOLIO_SITE_ROOT: PORTFOLIO_ROOT },
    )
    assert.notEqual(proc.status, 0)
    const err = JSON.parse(proc.stderr.trim())
    assert.equal(err.failure.code, 'match_not_over')
    assert.equal(proc.stdout.trim(), '')
  })

  it('derives outcome JSON for valid match-over fixture', () => {
    if (!HAS_PORTFOLIO) return
    const proc = runHarness(
      [join(FIXTURES, 'valid-match-over-seed42.json'), 'match-valid-seed42'],
      { ...process.env, PORTFOLIO_SITE_ROOT: PORTFOLIO_ROOT },
    )
    assert.equal(proc.status, 0, proc.stderr || proc.stdout)
    const outcome = JSON.parse(proc.stdout.trim())
    assert.equal(outcome.matchId, 'match-valid-seed42')
    assert.equal(outcome.createdAt, '2026-05-19T22:00:57.913Z')
    assert.equal(outcome.endVariant, 'victory')
    assert.equal(outcome.humanWon, true)
    assert.equal('seed' in outcome, false)
    assert.equal('version' in outcome, false)
  })

  for (const [fixtureName, expectedCode] of Object.entries(STRUCTURED_FAILURE_FIXTURES)) {
    it(`stderr failure code ${expectedCode} for ${fixtureName}`, () => {
      if (!HAS_PORTFOLIO) return
      const proc = runHarness([join(FIXTURES, fixtureName), `match-${fixtureName}`], {
        ...process.env,
        PORTFOLIO_SITE_ROOT: PORTFOLIO_ROOT,
      })
      assert.notEqual(proc.status, 0)
      const err = JSON.parse(proc.stderr.trim())
      assert.equal(err.failure.code, expectedCode)
      assert.equal(proc.stdout.trim(), '')
    })
  }

  it('resolves matchId from envelope when argv omits it', () => {
    if (!HAS_PORTFOLIO) return
    const envelopePath = resolveReplayEnvelope('victory')
    const envelope = JSON.parse(readFileSync(envelopePath, 'utf8'))
    assert.equal(typeof envelope.matchId, 'string')
    const proc = runHarness([envelopePath], {
      ...process.env,
      PORTFOLIO_SITE_ROOT: PORTFOLIO_ROOT,
    })
    assert.equal(proc.status, 0, proc.stderr || proc.stdout)
    const outcome = JSON.parse(proc.stdout.trim())
    assert.equal(outcome.matchId, envelope.matchId)
  })

  it('deep-equals golden using dungeon-runner outcome-parity envelope copy', () => {
    if (!HAS_PORTFOLIO) return
    const envelopePath = join(OUTCOME_PARITY, 'replay-envelope-outcome-victory.json')
    const golden = JSON.parse(readFileSync(resolveOutcomeGolden('victory'), 'utf8'))
    const proc = runHarness([envelopePath], {
      ...process.env,
      PORTFOLIO_SITE_ROOT: PORTFOLIO_ROOT,
    })
    assert.equal(proc.status, 0, proc.stderr || proc.stdout)
    assert.deepEqual(JSON.parse(proc.stdout.trim()), golden)
  })

  it('reads envelope from stdin when path is -', () => {
    if (!HAS_PORTFOLIO) return
    const envelope = readFileSync(join(FIXTURES, 'match-not-over.json'), 'utf8')
    const proc = spawnSync(process.execPath, [HARNESS, '-', 'match-not-over-empty'], {
      encoding: 'utf8',
      input: envelope,
      env: { ...process.env, PORTFOLIO_SITE_ROOT: PORTFOLIO_ROOT },
      cwd: REPO_ROOT,
    })
    assert.notEqual(proc.status, 0)
    assert.equal(JSON.parse(proc.stderr.trim()).failure.code, 'match_not_over')
  })

  for (const variant of PARITY_VARIANTS) {
    it(`deep-equals portfolio-site outcome golden for ${variant}`, () => {
      if (!HAS_PORTFOLIO) return
      const envelopePath = resolveReplayEnvelope(variant)
      const golden = JSON.parse(readFileSync(resolveOutcomeGolden(variant), 'utf8'))
      const proc = runHarness([envelopePath], {
        ...process.env,
        PORTFOLIO_SITE_ROOT: PORTFOLIO_ROOT,
      })
      assert.equal(proc.status, 0, proc.stderr || proc.stdout)
      const derived = JSON.parse(proc.stdout.trim())
      assert.deepEqual(derived, golden)
    })
  }
})
