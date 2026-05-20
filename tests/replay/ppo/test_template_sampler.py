"""Rollout match template sampler."""

from __future__ import annotations

import random

from dungeon_runner.replay.ppo.template_sampler import (
    TEMPLATE_BC_BOT,
    TEMPLATE_SELF_PLAY,
    TEMPLATE_VS_RANDOMBOT,
    _WEIGHTS,
    sample_rollout_template,
)


def test_default_template_mix_probabilities():
    assert _WEIGHTS == (0.20, 0.45, 0.35)


def test_sample_rollout_template_seeded_frequencies():
    rng = random.Random(7)
    counts = {TEMPLATE_VS_RANDOMBOT: 0, TEMPLATE_BC_BOT: 0, TEMPLATE_SELF_PLAY: 0}
    n = 10_000
    for _ in range(n):
        counts[sample_rollout_template(rng)] += 1
    assert 0.16 < counts[TEMPLATE_VS_RANDOMBOT] / n < 0.24
    assert 0.40 < counts[TEMPLATE_BC_BOT] / n < 0.50
    assert 0.30 < counts[TEMPLATE_SELF_PLAY] / n < 0.40


def test_sample_rollout_template_returns_known_ids():
    rng = random.Random(0)
    for _ in range(50):
        t = sample_rollout_template(rng)
        assert t in (TEMPLATE_VS_RANDOMBOT, TEMPLATE_BC_BOT, TEMPLATE_SELF_PLAY)
