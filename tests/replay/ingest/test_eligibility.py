"""Envelope eligibility (parity with portfolio-site importReplayEnvelope)."""

import pytest

from dungeon_runner.replay.eligibility import eligibility_skip_reason


def _valid(**extra):
    body = {
        "version": 1,
        "seed": 42,
        "setup": {"totalSeats": 2, "opponents": [{"type": "randombot"}]},
        "history": [],
    }
    body.update(extra)
    return body


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"version": 2},
        {"version": "1"},
        {"version": 1.5},
        {"version": True},
    ],
)
def test_unsupported_version(envelope):
    assert eligibility_skip_reason(envelope) == "unsupported_version"


def test_missing_seed():
    no_seed = _valid()
    del no_seed["seed"]
    assert eligibility_skip_reason(no_seed) == "missing_seed"
    assert eligibility_skip_reason(_valid(seed="42")) == "missing_seed"
    assert eligibility_skip_reason(_valid(seed=True)) == "missing_seed"


def test_missing_setup():
    assert eligibility_skip_reason(_valid(setup=None)) == "missing_setup"
    assert eligibility_skip_reason({k: v for k, v in _valid().items() if k != "setup"}) == "missing_setup"


def test_missing_history():
    assert eligibility_skip_reason(_valid(history=None)) == "missing_history"
    assert eligibility_skip_reason(_valid(history="[]")) == "missing_history"


def test_invalid_presentation_speed():
    assert (
        eligibility_skip_reason(_valid(presentationSpeedProfile="fast"))
        == "invalid_presentation_speed"
    )
    assert (
        eligibility_skip_reason(_valid(presentationSpeedProfile=None))
        == "invalid_presentation_speed"
    )


def test_accepts_valid_presentation_speed():
    assert eligibility_skip_reason(_valid(presentationSpeedProfile="brisk")) is None
    assert eligibility_skip_reason(_valid(presentationSpeedProfile="cinematic")) is None


def test_invalid_history_rng_steps():
    bad = _valid(
        history=[
            {
                "action": {"type": "PASS"},
                "actorSeatId": "seat-1",
                "rngStepBefore": 2,
                "rngStepAfter": 2,
            }
        ]
    )
    assert eligibility_skip_reason(bad) == "invalid_history"


def test_invalid_history_non_contiguous_chain():
    bad = _valid(
        history=[
            {
                "action": {"type": "PASS"},
                "actorSeatId": "seat-1",
                "rngStepBefore": 0,
                "rngStepAfter": 1,
            },
            {
                "action": {"type": "PASS"},
                "actorSeatId": "seat-2",
                "rngStepBefore": 3,
                "rngStepAfter": 4,
            },
        ]
    )
    assert eligibility_skip_reason(bad) == "invalid_history"


def test_accepts_empty_history():
    assert eligibility_skip_reason(_valid(history=[])) is None


def test_accepts_valid_history():
    ok = _valid(
        history=[
            {
                "action": {"type": "PASS"},
                "actorSeatId": "seat-1",
                "rngStepBefore": 0,
                "rngStepAfter": 1,
            }
        ]
    )
    assert eligibility_skip_reason(ok) is None


def test_preserves_unknown_keys_eligible():
    assert eligibility_skip_reason(_valid(rulesHash="abc", extra={"nested": True})) is None
