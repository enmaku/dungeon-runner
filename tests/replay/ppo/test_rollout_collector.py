"""Sim rollout collector: BC-bot vs RandomBot vs self-play seat roles."""

from __future__ import annotations

import pytest

from dungeon_runner.replay.bc.predict import load_policy_model
from dungeon_runner.replay.ppo.frozen_teacher import FrozenBCTeacher
from dungeon_runner.replay.ppo.rollout_collector import collect_rollouts_local
from dungeon_runner.replay.ppo.template_sampler import (
    TEMPLATE_BC_BOT,
    TEMPLATE_SELF_PLAY,
    TEMPLATE_VS_RANDOMBOT,
    roles_for_template,
)
from tests.replay.bc.bc_fixtures import PRODUCTION_PARENT_WEIGHTS


@pytest.fixture
def production_policy():
    if not PRODUCTION_PARENT_WEIGHTS.is_file():
        pytest.skip("models/latest/policy.weights.h5 not present")
    model = load_policy_model(PRODUCTION_PARENT_WEIGHTS)
    teacher = FrozenBCTeacher.from_weights(
        PRODUCTION_PARENT_WEIGHTS, load_model=load_policy_model
    )
    return model, teacher


def test_roles_for_template_single_learner_except_self_play():
    rng = __import__("random").Random(0)
    for template in (TEMPLATE_VS_RANDOMBOT, TEMPLATE_BC_BOT):
        roles = roles_for_template(template, 4, rng)
        assert sum(roles) == 1
    self_roles = roles_for_template(TEMPLATE_SELF_PLAY, 4, rng)
    assert self_roles == [True, True, True, True]


def test_vs_bc_bot_uses_teacher_on_opponent_seats(monkeypatch, production_policy):
    model, teacher = production_policy
    teacher_calls: list[int] = []
    orig = FrozenBCTeacher.select_masked

    def counting(self, obs, mask):
        teacher_calls.append(1)
        return orig(self, obs, mask)

    monkeypatch.setattr(FrozenBCTeacher, "select_masked", counting)
    monkeypatch.setattr(
        "dungeon_runner.replay.ppo.rollout_collector.sample_rollout_template",
        lambda _rng: TEMPLATE_BC_BOT,
    )

    batch, _stats, template = collect_rollouts_local(
        model, teacher, target_steps=12, seed=3
    )
    assert template == TEMPLATE_BC_BOT
    assert len(batch) >= 1
    assert len(teacher_calls) >= 1


def test_vs_randombot_does_not_call_teacher(monkeypatch, production_policy):
    model, teacher = production_policy
    teacher_calls: list[int] = []
    orig = FrozenBCTeacher.select_masked

    def counting(self, obs, mask):
        teacher_calls.append(1)
        return orig(self, obs, mask)

    monkeypatch.setattr(FrozenBCTeacher, "select_masked", counting)
    monkeypatch.setattr(
        "dungeon_runner.replay.ppo.rollout_collector.sample_rollout_template",
        lambda _rng: TEMPLATE_VS_RANDOMBOT,
    )

    batch, _stats, template = collect_rollouts_local(
        model, teacher, target_steps=12, seed=5
    )
    assert template == TEMPLATE_VS_RANDOMBOT
    assert len(batch) >= 1
    assert teacher_calls == []


def test_self_play_records_all_seats(monkeypatch, production_policy):
    model, teacher = production_policy
    monkeypatch.setattr(
        "dungeon_runner.replay.ppo.rollout_collector.sample_rollout_template",
        lambda _rng: TEMPLATE_SELF_PLAY,
    )
    batch, stats, template = collect_rollouts_local(
        model, teacher, target_steps=16, seed=11
    )
    assert template == TEMPLATE_SELF_PLAY
    assert len(batch) >= 2
    assert stats.env_steps >= len(batch)
