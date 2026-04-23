"""Optional pygame UI for ``Match`` state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dungeon_runner.ui.pygame_view import MatchViewConfig, PygameMatchView

if TYPE_CHECKING:
    from dungeon_runner.match import Match

__all__ = ["MatchViewConfig", "PygameMatchView", "MatchView", "get_match_view"]


@runtime_checkable
class MatchView(Protocol):
    @property
    def is_open(self) -> bool: ...
    def sync(self, m: "Match") -> None: ...
    def pump(self, duration_ms: float) -> bool: ...
    def show_static(self, m: "Match", message: str | None = None) -> bool: ...
    def close(self) -> None: ...


def get_match_view(
    config: MatchViewConfig | None = None,
) -> PygameMatchView:
    return PygameMatchView(config or MatchViewConfig())
