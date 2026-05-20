"""Ingest eligibility checks aligned with portfolio-site importReplayEnvelope."""

from __future__ import annotations

from typing import Any


def _is_int(value: object) -> bool:
    return type(value) is int


def _history_skip_reason(history: list[Any]) -> str | None:
    previous_after: int | None = None
    for entry in history:
        if not isinstance(entry, dict):
            return "invalid_history"
        action = entry.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            return "invalid_history"
        actor = entry.get("actorSeatId")
        if not isinstance(actor, str) or not actor:
            return "invalid_history"
        before = entry.get("rngStepBefore")
        after = entry.get("rngStepAfter")
        if not _is_int(before) or not _is_int(after):
            return "invalid_history"
        if after <= before:
            return "invalid_history"
        if previous_after is not None and before != previous_after:
            return "invalid_history"
        previous_after = after
    return None


def eligibility_skip_reason(envelope: dict[str, Any]) -> str | None:
    version = envelope.get("version")
    if not _is_int(version) or version != 1:
        return "unsupported_version"

    seed = envelope.get("seed")
    if not _is_int(seed):
        return "missing_seed"

    setup = envelope.get("setup")
    if not isinstance(setup, dict):
        return "missing_setup"

    history = envelope.get("history")
    if not isinstance(history, list):
        return "missing_history"

    if "presentationSpeedProfile" in envelope:
        pace = envelope["presentationSpeedProfile"]
        if pace != "cinematic" and pace != "brisk":
            return "invalid_presentation_speed"

    return _history_skip_reason(history)

