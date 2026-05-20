"""Eligibility parity with portfolio-site importReplayEnvelope tests."""

import pytest

from dungeon_runner.replay.eligibility import eligibility_skip_reason

_BASE = {
    "version": 1,
    "seed": 0,
    "setup": {"totalSeats": 2, "opponents": [{"type": "randombot"}]},
    "history": [],
}


@pytest.mark.parametrize(
    "envelope,reason",
    [
        ({**_BASE, "version": 1, "seed": None}, "missing_seed"),
        ({**_BASE, "seed": "x"}, "missing_seed"),
        ({**_BASE, "presentationSpeedProfile": "fast"}, "invalid_presentation_speed"),
        ({**_BASE, "presentationSpeedProfile": None}, "invalid_presentation_speed"),
        (
            {
                **_BASE,
                "history": [
                    {
                        "action": {"type": "PASS"},
                        "actorSeatId": "seat-1",
                        "rngStepBefore": 2,
                        "rngStepAfter": 2,
                    }
                ],
            },
            "invalid_history",
        ),
        (
            {
                **_BASE,
                "history": [
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
                ],
            },
            "invalid_history",
        ),
        ({**_BASE, "version": "1"}, "unsupported_version"),
        ({**_BASE, "version": 1.5}, "unsupported_version"),
    ],
)
def test_parity_skip_reasons_match_web_import(envelope, reason):
    assert eligibility_skip_reason(envelope) == reason


@pytest.mark.parametrize(
    "envelope",
    [
        _BASE,
        {**_BASE, "presentationSpeedProfile": "brisk"},
        {**_BASE, "presentationSpeedProfile": "cinematic"},
        {**_BASE, "rulesHash": "abc", "extra": {"nested": True}},
        {
            **_BASE,
            "seed": 7,
            "history": [
                {
                    "action": {"type": "PASS"},
                    "actorSeatId": "seat-1",
                    "rngStepBefore": 0,
                    "rngStepAfter": 1,
                }
            ],
        },
    ],
)
def test_parity_accepts_web_import_ok_cases(envelope):
    assert eligibility_skip_reason(envelope) is None


def test_parity_malformed_missing_seed_matches_web_invalid_replay():
    body = {
        "version": 1,
        "setup": {"totalSeats": 2, "opponents": [{"type": "randombot"}]},
        "history": [],
    }
    assert eligibility_skip_reason(body) == "missing_seed"
