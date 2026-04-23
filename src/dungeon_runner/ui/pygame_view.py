"""Pygame view styled like a physical table: facedown cards, equipment row, and motion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dungeon_runner.match import Match

from dungeon_runner.types_core import AdventurerKind

_COLOR_BG = (24, 28, 32)
_COLOR_FELT = (28, 52, 40)
_COLOR_PANEL = (45, 50, 58)
_COLOR_ACCENT = (120, 170, 220)
_COLOR_TEXT = (230, 232, 235)
_COLOR_DIM = (120, 125, 135)
_COLOR_CARD = (70, 76, 88)
_COLOR_HP = (90, 200, 130)
_COLOR_FLASH = (230, 80, 80)
_COLOR_WARN = (230, 200, 90)
_COLOR_DECK = (55, 62, 75)
_COLOR_SPENT = (50, 48, 55)
_COLOR_SAC = (80, 45, 50)


def _hero_theme(h: AdventurerKind) -> dict[str, tuple[int, int, int]]:
    """Tabletop-style color for adventurer + matching gear. Warrior blue, Barbarian red, Mage purple, Rogue green."""
    base = {
        AdventurerKind.WARRIOR: (38, 70, 118),
        AdventurerKind.BARBARIAN: (130, 42, 48),
        AdventurerKind.MAGE: (90, 52, 128),
        AdventurerKind.ROGUE: (40, 118, 75),
    }[h]
    a0, a1, a2 = base

    def _clamp(x: int) -> int:
        return max(0, min(255, x))

    active = (
        _clamp(int(a0 * 1.15) + 18),
        _clamp(int(a1 * 1.12) + 18),
        _clamp(int(a2 * 1.12) + 22),
    )
    hand = (
        _clamp((a0 * 2 + 35) // 3),
        _clamp((a1 * 2 + 40) // 3),
        _clamp((a2 * 2 + 48) // 3),
    )
    panel = (
        _clamp((a0 + 22) // 2 + 6),
        _clamp((a1 + 24) // 2 + 8),
        _clamp((a2 + 28) // 2 + 6),
    )
    idle = (
        _clamp((a0 * 2 + 48) // 3),
        _clamp((a1 * 2 + 52) // 3),
        _clamp((a2 * 2 + 55) // 3),
    )
    sac = (
        _clamp((a0 + 95) // 2),
        _clamp((a1 + 50) // 2),
        _clamp((a2 + 50) // 2),
    )
    spent = (
        _clamp(a0 // 2 + 20),
        _clamp(a1 // 2 + 20),
        _clamp(a2 // 2 + 20),
    )
    border = (
        _clamp(a0 + 25),
        _clamp(a1 + 25),
        _clamp(a2 + 30),
    )
    return {
        "primary": base,
        "active": active,
        "hand": hand,
        "panel": panel,
        "idle": idle,
        "sac": sac,
        "spent": spent,
        "border": border,
    }


@dataclass
class MatchViewConfig:
    width: int = 1100
    height: int = 720
    font_size: int = 20
    small_font_size: int = 15
    # Human pacing: defaults are slow; use CLI to speed up
    step_delay_ms: float = 1400.0
    dungeon_step_delay_ms: float = 2000.0
    end_screen_ms: float = 8000.0
    flash_duration_ms: float = 220.0
    # If True, show next deck card and more debug. Table view keeps secrets.
    god_mode: bool = False
    # Layout / animation
    table_mode: bool = True
    anim_deck_to_hand_ms: float = 600.0
    anim_hand_to_pile_ms: float = 650.0
    anim_sacrifice_ms: float = 500.0
    anim_dungeon_reveal_ms: float = 450.0
    # Big banner when a dungeon run ends (Success / Failure)
    run_outcome_banner_ms: float = 5000.0
    title: str = "Welcome to the Dungeon"


@dataclass
class _Flash:
    until_tick: int = 0
    kind: str = ""  # "hp" | "monster" | ""


@dataclass
class _ActiveAnim:
    kind: str
    t0: int
    duration: int
    # flying card: label lines
    line1: str
    line2: str
    # start/end in px (x,y, w, h) for lerp
    x0: float
    y0: float
    x1: float
    y1: float
    w: int
    h: int
    # optional equipment slot index for sac (0-5) — draw from there
    equip_index: int = -1
    # hide last n cards from displayed pile while flying (1 for hand->pile)
    hide_pile_top: int = 0
    # dungeon reveal pulse
    d_def_id: int = -1


@dataclass
class PygameMatchView:
    config: MatchViewConfig
    _screen: object = field(default=None, repr=False)
    _font: object = field(default=None, repr=False)
    _font_lg: object = field(default=None, repr=False)
    _font_small: object = field(default=None, repr=False)
    _clock: object = field(default=None, repr=False)
    _open: bool = field(default=True, init=False)
    _prev_hp: Optional[int] = field(default=None, init=False)
    _prev_d_current_id: Optional[tuple[Optional[int], str]] = field(
        default=None, init=False
    )
    _flash: _Flash = field(default_factory=_Flash, init=False)
    _last_m: Optional["Match"] = field(default=None, init=False, repr=False)
    _active_anim: Optional[_ActiveAnim] = field(default=None, init=False)
    _post_anim_pump_end: int = 0
    _pygame: object = field(default=None, repr=False)
    _font_xl: object = field(default=None, repr=False)
    _run_outcome: str = field(default="", init=False)  # "success" | "failure" | ""
    _run_outcome_detail: str = field(default="", init=False)
    _run_outcome_until: int = field(default=0, init=False)
    _table_top: int = field(default=200, init=False)  # y of card row; keep in sync for animations

    def __post_init__(self) -> None:
        self._init_pygame()

    def _init_pygame(self) -> None:
        import pygame  # type: ignore[import-untyped]

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.config.width, self.config.height)
        )
        pygame.display.set_caption(self.config.title)
        self._font = pygame.font.Font(None, self.config.font_size)
        self._font_lg = pygame.font.Font(None, int(self.config.font_size * 1.35))
        self._font_xl = pygame.font.Font(None, int(self.config.font_size * 2.4))
        self._font_small = pygame.font.Font(None, self.config.small_font_size)
        self._clock = pygame.time.Clock()

    @property
    def pygame(self) -> object:
        return self._pygame

    @property
    def is_open(self) -> bool:
        return self._open

    def close(self) -> None:
        if not self._open or self._pygame is None:
            return
        self._pygame.display.quit()  # type: ignore[union-attr]
        self._pygame.quit()
        self._open = False

    def _lerp(self, t: float) -> float:
        # smoothstep
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _layout(self) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
        """Deck, hand, pile, sac, stage — uses ``self._table_top`` (set in ``_draw`` each frame)."""
        return self._layout_at(self._table_top)

    def _y_table_top(self, m: "Match") -> int:
        from dungeon_runner.match import MatchPhase, MatchTerminalReason

        y = 44
        phase_t = m.phase.name.replace("_", " ")
        t = f"Active seat: {m.active_seat}  ·  {phase_t}"
        y += int(self._font.render(t, True, (0, 0, 0)).get_height())  # type: ignore[union-attr, no-untyped-call]
        y += 8
        y += 32
        y += 6
        add_bits: list[str] = []
        for i, pl in enumerate(m.players):
            adds = [x.value for x in pl.own_pile_adds]
            if adds:
                add_bits.append(f"P{i} -> {', '.join(adds)}")
        if add_bits:
            ts = "You remember your adds: " + "  ·  ".join(add_bits)[:220]
            y += int(self._font_small.render(ts, True, (0, 0, 0)).get_height())  # type: ignore[union-attr, no-untyped-call]
        y += 10
        if m.phase is MatchPhase.ENDED and m.winner_seat is not None:
            tr = m.terminal_reason
            tname = tr.name if isinstance(tr, MatchTerminalReason) else "?"
            t2 = f"Match over  -  winner: seat {m.winner_seat} ({tname})"
            y += int(self._font.render(t2, True, (0, 0, 0)).get_height())  # type: ignore[union-attr, no-untyped-call]
            y += 28
        if m.phase is MatchPhase.DUNGEON and m.runner_seat is not None:
            y += int(
                self._font.render(  # type: ignore[no-untyped-call, union-attr]
                    f"Runner this dungeon: P{m.runner_seat}", True, (0, 0, 0)
                ).get_height()  # type: ignore[no-untyped-call, union-attr]
            )
            y += 24
        if m.phase is MatchPhase.PICK_ADVENTURER and m.pick_next_seat is not None:
            t3 = f"Choose next hero: P{m.pick_next_seat} picks from Warrior / Barbarian / Mage / Rogue"
            y += int(self._font.render(t3, True, (0, 0, 0)).get_height())  # type: ignore[no-untyped-call, union-attr]
            y += 24
        y += 8
        return y + 6

    def _wrap_to_width(self, text: str, font: object, max_w: int) -> list[str]:
        words = (text or "").replace("\u2014", "-").replace("\u2013", "-").split()
        if not words:
            return []
        out: list[str] = []
        cur: list[str] = []
        for w in words:
            trial = " ".join([*cur, w]) if cur else w
            tw: int = int(
                font.size(trial)[0]  # type: ignore[union-attr, no-untyped-call]
            )
            if (tw <= max_w or not cur) and trial.strip():
                cur.append(w)
            else:
                if cur:
                    out.append(" ".join(cur))
                cur = [w]
        if cur:
            out.append(" ".join(cur))
        return out

    def _layout_at(self, table_y: int) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
        w, h = self.config.width, self.config.height
        m = 28
        row_h = 110
        # Must match equipment row + label offset in _draw; keeps stage from covering them.
        res_bottom = 200
        deck = (m, table_y, 100, row_h)
        hand = (w // 2 - 60, table_y, 120, row_h)
        pile = (w - 224, table_y, 198, 152)
        sac = (w - 224, table_y + 156, 198, 64)
        gap = 16
        y_stage = table_y + row_h + gap
        st_h = h - y_stage - res_bottom
        st_h = max(100, st_h)
        if y_stage + st_h > h - res_bottom:
            st_h = h - y_stage - res_bottom
        st_h = max(100, st_h)
        stage = (m, y_stage, w - 2 * m, st_h)
        return (*deck, *hand, *pile, *sac, *stage)

    def _draw_back(self, surf: object, r: tuple[int, int, int, int], label: str = "") -> None:
        pygame = self._pygame
        pygame.draw.rect(surf, _COLOR_DECK, r)  # type: ignore[union-attr]
        pygame.draw.rect(surf, (90, 95, 110), r, 2)  # type: ignore[union-attr]
        if label:
            img = self._font_small.render(label, True, _COLOR_DIM)  # type: ignore[union-attr, no-untyped-call]
            surf.blit(img, (r[0] + 6, r[1] + 4))  # type: ignore[union-attr]

    def _draw_flying_card(
        self,
        surf: object,
        a: _ActiveAnim,
        now: int,
    ) -> None:
        t = (now - a.t0) / max(1, a.duration)
        u = self._lerp(t)
        x = a.x0 + (a.x1 - a.x0) * u
        y = a.y0 + (a.y1 - a.y0) * u
        r = (int(x), int(y), a.w, a.h)
        pygame = self._pygame
        pygame.draw.rect(surf, _COLOR_CARD, r)  # type: ignore[union-attr]
        pygame.draw.rect(surf, _COLOR_ACCENT, r, 3)  # type: ignore[union-attr]
        img1 = self._font.render(a.line1, True, _COLOR_TEXT)  # type: ignore[union-attr, no-untyped-call]
        img2 = self._font_small.render(a.line2, True, _COLOR_DIM)  # type: ignore[union-attr, no-untyped-call]
        surf.blit(img1, (r[0] + 8, r[1] + 12))  # type: ignore[union-attr]
        surf.blit(img2, (r[0] + 8, r[1] + 38))  # type: ignore[union-attr]

    def _detect_animation(self, old: Optional["Match"], m: "Match") -> None:
        if old is None or self._active_anim is not None:
            return
        from dungeon_runner.match import BiddingState, MatchPhase

        dx, dy, dw, dh, hx, hy, hw, hh, px, py, pw, ph, sx, sy, sw, sh, stx, sty, stw, sth = self._layout()

        if m.phase is MatchPhase.BIDDING and old.phase is MatchPhase.BIDDING:
            if (
                len(m.monster_deck) < len(old.monster_deck)
                and m.bidding_sub is BiddingState.PENDING
                and m.pending_card is not None
            ):
                pc = m.pending_card
                pg = self._pygame
                self._active_anim = _ActiveAnim(
                    "deck_to_hand",
                    int(pg.time.get_ticks()),  # type: ignore[union-attr]
                    int(self.config.anim_deck_to_hand_ms),
                    pc.species.value,
                    f"str {pc.strength}",
                    float(dx + dw // 2 - 50),
                    float(dy + dh // 2 - 40),
                    float(hx + hw // 2 - 50),
                    float(hy + hh // 2 - 40),
                    100,
                    88,
                )
                return
            if (
                len(m.dungeon_pile) > len(old.dungeon_pile)
                and old.bidding_sub is BiddingState.PENDING
            ):
                card = m.dungeon_pile[-1]
                pg = self._pygame
                self._active_anim = _ActiveAnim(
                    "hand_to_pile",
                    int(pg.time.get_ticks()),  # type: ignore[union-attr]
                    int(self.config.anim_hand_to_pile_ms),
                    card.species.value,
                    f"str {card.strength}",
                    float(hx + hw // 2 - 50),
                    float(hy + hh // 2 - 40),
                    float(px + pw // 2 - 50),
                    float(py + ph // 2 - 40),
                    100,
                    88,
                    hide_pile_top=1,
                )
                return
            if len(m.sacrifice_rows) > len(old.sacrifice_rows) and m.sacrifice_rows:
                row = m.sacrifice_rows[-1]
                from dungeon_runner.catalog import HERO_LOADOUT, ALL_EQUIP_DB

                order = list(HERO_LOADOUT[m.hero].equipment_ids)
                eid = row.equipment_id
                idx = order.index(eid) if eid in order else -1
                yb = self.config.height - 162
                eqx = 20 + (idx % 3) * 200
                eqy = yb + (max(0, idx) // 3) * 50
                pg = self._pygame
                self._active_anim = _ActiveAnim(
                    "sacrifice",
                    int(pg.time.get_ticks()),  # type: ignore[union-attr]
                    int(self.config.anim_sacrifice_ms),
                    ALL_EQUIP_DB.get(eid, (eid, 0, 0, None))[0][:14],
                    "to discard",
                    float(eqx + 20),
                    float(eqy - 8),
                    float(sx + sw // 2 - 40),
                    float(sy + 20),
                    88,
                    56,
                    equip_index=idx,
                )
                return

        if m.phase is MatchPhase.DUNGEON and old.phase is MatchPhase.DUNGEON:
            if m.d_current and (
                old.d_current is None or m.d_current.def_id != old.d_current.def_id
            ):
                c = m.d_current
                pg = self._pygame
                self._active_anim = _ActiveAnim(
                    "dungeon_reveal",
                    int(pg.time.get_ticks()),  # type: ignore[union-attr]
                    int(self.config.anim_dungeon_reveal_ms),
                    c.species.value,
                    f"str {c.strength} · new reveal",
                    float(stx + stw // 2 - 2),
                    float(sty - 2),
                    float(stx + stw // 2 - 2),
                    float(sty - 2),
                    min(200, stw - 8),
                    min(120, sth - 8),
                    d_def_id=c.def_id,
                )
                return

    def _player_status_line(self, m: "Match", surf: object, y: int) -> int:
        from dungeon_runner.match import MatchPhase

        # ASCII + colors — default SysFont does not render U+2705, U+2713, U+23F1, etc.
        x = 20
        show_turn = m.phase is not MatchPhase.ENDED
        leg = self._font_small.render(  # type: ignore[no-untyped-call, union-attr]
            "Mark:  [+] success  [-] hurt  [p] pass (bid)  [OUT] out  |  box = turn  |  ",
            True,
            _COLOR_DIM,
        )
        surf.blit(leg, (x, y))  # type: ignore[union-attr]
        x += leg.get_width()  # type: ignore[union-attr, no-untyped-call]
        pygame = self._pygame
        for i, pl in enumerate(m.players):
            box = f"P{i} "
            on_turn = show_turn and i == m.active_seat
            c = _COLOR_WARN if on_turn else _COLOR_TEXT
            img0 = self._font.render(box, True, c)  # type: ignore[no-untyped-call, union-attr]
            tw = int(img0.get_width())  # type: ignore[no-untyped-call, union-attr]
            th = int(img0.get_height())  # type: ignore[no-untyped-call, union-attr]
            if on_turn:
                padx, pady = 2, 1
                br = (x - padx, y - 1 - pady, tw + 2 * padx, th + 2 * pady + 1)
                pygame.draw.rect(surf, (32, 58, 72), br)  # type: ignore[union-attr]
                pygame.draw.rect(surf, (210, 185, 75), br, 2)  # type: ignore[union-attr]
            surf.blit(img0, (x, y))  # type: ignore[union-attr]
            x += tw  # type: ignore[no-untyped-call, union-attr]
            if pl.eliminated:
                s = self._font.render("[OUT]", True, (255, 90, 90))  # type: ignore[no-untyped-call, union-attr]
                surf.blit(s, (x, y - 1))  # type: ignore[union-attr]
                x += s.get_width() + 4  # type: ignore[union-attr, no-untyped-call]
            for _ in range(pl.success_cards):
                s = self._font.render("[+]", True, (100, 220, 120))  # type: ignore[no-untyped-call, union-attr]
                surf.blit(s, (x, y - 1))  # type: ignore[union-attr]
                x += s.get_width() + 1  # type: ignore[union-attr, no-untyped-call]
            for _ in range(min(2, pl.aid_flips)):
                s = self._font.render("[-]", True, (255, 100, 100))  # type: ignore[no-untyped-call, union-attr]
                surf.blit(s, (x, y - 1))  # type: ignore[union-attr]
                x += s.get_width() + 1  # type: ignore[union-attr, no-untyped-call]
            if m.phase is MatchPhase.BIDDING and pl.has_passed_bid:
                s = self._font.render("[p]", True, (255, 200, 120))  # type: ignore[no-untyped-call, union-attr]
                surf.blit(s, (x, y - 1))  # type: ignore[union-attr]
                x += s.get_width() + 4  # type: ignore[union-attr, no-untyped-call]
            x += 12
        return 32

    def _draw_run_outcome_banner(self, surf: object, now: int) -> None:
        if not self._run_outcome or now >= self._run_outcome_until:
            return
        w, h = self.config.width, self.config.height
        pygame = self._pygame
        band_h = 118
        by = h // 2 - 100
        if self._run_outcome == "success":
            bg = (32, 78, 48)
            title = "DUNGEON CLEARED"
            tc = (180, 255, 200)
        else:
            bg = (88, 35, 40)
            title = "DUNGEON LOST"
            tc = (255, 200, 195)
        pygame.draw.rect(surf, bg, (0, by, w, band_h))  # type: ignore[union-attr]
        pygame.draw.rect(surf, (255, 255, 255), (0, by, w, band_h), 2)  # type: ignore[union-attr]
        t1 = self._font_xl.render(title, True, tc)  # type: ignore[no-untyped-call, union-attr]
        tw = t1.get_width()  # type: ignore[union-attr, no-untyped-call]
        surf.blit(t1, ((w - tw) // 2, by + 8))  # type: ignore[union-attr]
        if self._run_outcome_detail:
            t2 = self._font.render(self._run_outcome_detail[:100], True, (240, 240, 245))  # type: ignore[no-untyped-call, union-attr]
            t2w = t2.get_width()  # type: ignore[union-attr, no-untyped-call]
            surf.blit(t2, ((w - t2w) // 2, by + 70))  # type: ignore[union-attr]

    def _disposition(self, m: "Match") -> str:
        log = m.dungeon_run_log
        if not log:
            return ""
        for line in reversed(log[-4:]):
            t = line.strip()
            if not t:
                continue
            if t.startswith("Reveal "):
                return t
            if t.startswith("  ") and len(t) > 3:
                return t[:100]
        return log[-1].strip()[:100]

    def _draw_label(
        self, surf: object, x: int, y: int, text: str, color: tuple[int, int, int] | None = None
    ) -> int:
        c = color or _COLOR_TEXT
        img = self._font.render(text, True, c)
        surf.blit(img, (x, y))  # type: ignore[union-attr]
        return img.get_height()  # type: ignore[union-attr, no-untyped-call]

    def _draw_label_small(
        self, surf: object, x: int, y: int, text: str, color: tuple[int, int, int] | None = None
    ) -> int:
        c = color or _COLOR_TEXT
        img = self._font_small.render(text, True, c)
        surf.blit(img, (x, y))  # type: ignore[union-attr]
        return img.get_height()  # type: ignore[union-attr, no-untyped-call]

    def _draw(
        self,
        m: "Match",
        *,
        overlay: str | None = None,
        now_override: int | None = None,
    ) -> None:
        from dungeon_runner.catalog import ALL_EQUIP_DB, HERO_LOADOUT
        from dungeon_runner.match import BiddingState, MatchPhase, MatchTerminalReason

        pygame = self._pygame
        surf = self._screen
        now = int(now_override if now_override is not None else pygame.time.get_ticks())  # type: ignore[union-attr]
        fl_active = self._flash.until_tick > now
        a = self._active_anim
        t_anim = 0.0
        if a and a.duration:
            t_anim = min(1.0, max(0.0, (now - a.t0) / float(a.duration)))
        lerp = self._lerp(t_anim)

        surf.fill(_COLOR_FELT)  # type: ignore[union-attr]
        self._rect_header(surf, 12, 6, f"{self.config.title}  (table view)", self.config.width - 24, 32)

        theme = _hero_theme(m.hero)
        y = 44
        phase_t = m.phase.name.replace("_", " ")
        y += self._draw_label(
            surf, 20, y, f"Active seat: {m.active_seat}  ·  {phase_t}", _COLOR_ACCENT
        )
        y += 8
        y += self._player_status_line(m, surf, y)
        y += 6
        add_bits: list[str] = []
        for i, pl in enumerate(m.players):
            adds = [x.value for x in pl.own_pile_adds]
            if adds:
                add_bits.append(f"P{i} -> {', '.join(adds)}")
        if add_bits:
            y += self._draw_label_small(
                surf, 20, y, "You remember your adds: " + "  ·  ".join(add_bits)[:220], (150, 155, 160)
            )
        y += 10
        if m.phase is MatchPhase.ENDED and m.winner_seat is not None:
            tr = m.terminal_reason
            tname = tr.name if isinstance(tr, MatchTerminalReason) else "?"
            y += self._draw_label(
                surf, 20, y, f"Match over  -  winner: seat {m.winner_seat} ({tname})",
                _COLOR_WARN,
            )
            y += 28
        if m.phase is MatchPhase.DUNGEON and m.runner_seat is not None:
            y += self._draw_label(
                surf, 20, y, f"Runner this dungeon: P{m.runner_seat}", _COLOR_WARN
            )
            y += 24
        if m.phase is MatchPhase.PICK_ADVENTURER and m.pick_next_seat is not None:
            y += self._draw_label(
                surf, 20, y, f"Choose next hero: P{m.pick_next_seat} picks from Warrior / Barbarian / Mage / Rogue",
                _COLOR_ACCENT,
            )
            y += 24
        y += 8
        self._table_top = self._y_table_top(m)
        # --- Table zones (all rects derived from this y so nothing sits under text above) ---
        dx, dy, dw, dh, hx, hy, hw, hh, px, py, pw, ph, sx, sy, sw, sh, stx, sty, stw, sth = self._layout()

        self._draw_back(surf, (dx, dy, dw, dh), f"Deck  {len(m.monster_deck)}")

        if m.phase is MatchPhase.BIDDING and m.monster_deck and self.config.god_mode:
            top = m.monster_deck[0]
            self._draw_label_small(
                surf, dx + 6, dy + 52, f"(debug: {top.species.value})", (100, 140, 100)
            )
        n_show_pile = 0
        if m.phase is MatchPhase.DUNGEON:
            n_show_pile = len(m.d_remaining)
        elif m.phase is MatchPhase.BIDDING:
            n_show_pile = len(m.dungeon_pile)
        if a and a.hide_pile_top and a.kind == "hand_to_pile" and t_anim < 0.99:
            n_show_pile = max(0, n_show_pile - 1)
        stack = min(8, max(0, n_show_pile))
        for i in range(stack):
            o = i * 4
            self._draw_back(
                surf,
                (px + 12 + o, py + 20 + o // 2, 64, 88),
                "",
            )
        n_label = n_show_pile
        if m.phase is MatchPhase.DUNGEON and m.d_current is not None:
            n_label = len(m.d_remaining)  # not counting current face-up
        elif m.phase is MatchPhase.DUNGEON:
            n_label = len(m.d_remaining)
        n_txt = self._font_lg.render(str(n_label), True, (255, 255, 255))  # type: ignore[no-untyped-call, union-attr]
        cw = n_txt.get_width()  # type: ignore[union-attr, no-untyped-call]
        ch = n_txt.get_height()  # type: ignore[union-attr, no-untyped-call]
        surf.blit(n_txt, (px + (pw - cw) // 2, py + 42))  # type: ignore[union-attr]
        if m.phase is MatchPhase.DUNGEON or m.phase is MatchPhase.BIDDING:
            cap = "in pile (bidding: building)" if m.phase is MatchPhase.BIDDING else "facedown left to reveal"
            self._draw_label_small(
                surf, px + 6, py + 40 + ch + 4, cap, _COLOR_DIM
            )
        if stack == 0 and m.phase is MatchPhase.BIDDING:
            self._draw_label_small(surf, px + 8, py + 56, "empty", _COLOR_DIM)
        # Hand zone (tinted to this hero, like a player mat)
        pygame.draw.rect(  # type: ignore[union-attr]
            surf, theme["hand"], (hx, hy, hw, hh)
        )
        pygame.draw.rect(surf, theme["border"], (hx, hy, hw, hh), 2)  # type: ignore[union-attr]
        self._draw_label_small(
            surf, hx + 8, hy + 4, "Your draw (facedown)", _COLOR_DIM
        )
        if m.bidding_sub is BiddingState.PENDING and m.pending_card and m.phase is MatchPhase.BIDDING:
            if self.config.god_mode or not self.config.table_mode:
                c = m.pending_card
                self._rect_titled(
                    surf,
                    (hx + 4, hy + 22, hw - 8, 78),
                    f"{c.species.value[:12]}\nstr {c.strength}",
                    border_rgb=theme["border"],
                )
            else:
                self._draw_back(
                    surf,
                    (hx + 6, hy + 22, hw - 12, 70),
                    "Card",
                )

        # Side pile label (count in header; big number is drawn above)
        self._draw_label_small(
            surf,
            px + 6,
            py + 4,
            f"Dungeon pile  ({n_label} cards)  -  shuffled, facedown",
            _COLOR_DIM,
        )
        if m.sacrifice_rows:
            self._draw_label_small(
                surf, sx + 4, sy + 4, f"Sacrifices ({len(m.sacrifice_rows)})  facedown", _COLOR_DIM
            )
            for i, _ in enumerate(m.sacrifice_rows[:3]):
                self._draw_back(surf, (sx + 8 + i * 6, sy + 30 + i, 50, 42), "")

        if m.phase is MatchPhase.DUNGEON:
            hp_c = _COLOR_HP
            if fl_active and self._flash.kind == "hp":
                hp_c = _COLOR_FLASH
            pulse = 0.0
            if a and a.kind == "dungeon_reveal" and m.d_current and m.d_current.def_id == a.d_def_id:
                pulse = 0.08 * math.sin(t_anim * math.pi)
            bx = stx - int(pulse * 8)
            by = sty - int(pulse * 4)
            bw = stw + int(pulse * 16)
            bh = sth + int(pulse * 8)
            mid = bx + int(bw * 0.48)
            log_x = mid + 10
            log_w = bw - (log_x - bx) - 14
            pygame.draw.rect(  # type: ignore[union-attr]
                surf, theme["panel"], (bx, by, bw, bh)
            )
            pygame.draw.rect(surf, theme["border"], (bx, by, bw, bh), 4)  # type: ignore[union-attr]
            title_h = self._font_lg.render("Dungeon run", True, (250, 250, 252))  # type: ignore[no-untyped-call, union-attr]
            th = int(title_h.get_height())  # type: ignore[no-untyped-call, union-attr]
            t_top = 8
            surf.blit(title_h, (bx + 16, by + t_top))  # type: ignore[union-attr]
            pygame.draw.line(  # type: ignore[union-attr]
                surf,
                (100, 110, 120),
                (mid, by + t_top + th + 4),
                (mid, by + bh - 8),
                1,
            )
            yc = by + t_top + th + 12
            yc += self._draw_label_lg(surf, bx + 14, yc, f"HP  {m.d_hp}", hp_c)
            dcur = m.d_current
            m_col = theme["active"]
            if fl_active and self._flash.kind == "monster":
                m_col = (255, 230, 120)
            if m.dungeon_sub and m.dungeon_sub.name in (
                "VORPAL", "PICK_FIRE_AXE", "PICK_POLYMORPH", "REVEAL"
            ):
                yc += 4
                yc += self._draw_label_small(
                    surf, bx + 14, yc, f"Choice: {m.dungeon_sub.name.replace('_', ' ').lower()}", _COLOR_WARN
                )
            yc += 6
            if dcur is not None:
                yc += 2
                yc += self._draw_label_lg(
                    surf, bx + 14, yc, dcur.species.value, m_col
                )
                yc += 2
                yc += self._draw_label(
                    surf,
                    bx + 14,
                    yc,
                    f"Strength {dcur.strength}  |  {len(m.d_remaining)} in pile (facedown)",
                    _COLOR_DIM,
                )
                yc += 6
                disp = self._disposition(m)
                if disp:
                    yc += self._draw_label(
                        surf, bx + 14, yc, f">> {disp}", (200, 220, 200)
                    )
            ltitle = self._font.render("Event log", True, (230, 235, 240))  # type: ignore[no-untyped-call, union-attr]
            surf.blit(ltitle, (log_x, by + t_top))  # type: ignore[union-attr]
            lth = int(ltitle.get_height())  # type: ignore[no-untyped-call, union-attr]
            lyy = by + t_top + lth + 4
            dlog = m.dungeon_run_log
            log_lines: list[str] = []
            for row in dlog[-12:]:
                for ln in self._wrap_to_width(
                    (row or "").replace("\u2014", "-").replace("\u2013", "-"),
                    self._font,
                    log_w,
                ):
                    log_lines.append(ln)
            line_h = int(self._font.get_height())  # type: ignore[no-untyped-call, union-attr]
            for ln in log_lines[- int((bh - 50) // max(16, line_h)) :]:
                if lyy + line_h > by + bh - 6:
                    break
                img = self._font.render(ln, True, (200, 205, 210))  # type: ignore[no-untyped-call, union-attr]
                surf.blit(img, (log_x, lyy))  # type: ignore[union-attr]
                lyy += line_h
        else:
            pygame.draw.rect(  # type: ignore[union-attr]
                surf, theme["idle"], (stx, sty, stw, sth)
            )
            pygame.draw.rect(  # type: ignore[union-attr]
                surf, theme["border"], (stx, sty, stw, sth), 2
            )
            idle_msg = "Bidding & setup  -  no dungeon run yet"
            img = self._font_lg.render(idle_msg, True, (240, 242, 245))  # type: ignore[no-untyped-call, union-attr]
            ix = stx + (stw - img.get_width()) // 2  # type: ignore[no-untyped-call, union-attr]
            iy = sty + (sth - img.get_height()) // 2  # type: ignore[no-untyped-call, union-attr]
            surf.blit(img, (ix, iy))  # type: ignore[union-attr]

        # Equipment: fixed 2 x 3 grid for 6 hero tiles
        y_eq = self.config.height - 162
        y += 0
        self._draw_label(
            surf, 20, y_eq - 22, "Adventurer equipment (X = sacrificed to dodge a card, or used up this run)", _COLOR_DIM
        )
        order = (
            list(HERO_LOADOUT[m.hero].equipment_ids)
            if m.phase
            in (
                MatchPhase.BIDDING,
                MatchPhase.DUNGEON,
                MatchPhase.ENDED,
                MatchPhase.PICK_ADVENTURER,
            )
            else []
        )
        sac_set = {r.equipment_id for r in m.sacrifice_rows}
        for i, eid in enumerate(order):
            ex = 20 + (i % 3) * 200
            ey = y_eq + (i // 3) * 50
            name = ALL_EQUIP_DB.get(eid, (eid, 0, 0, None))[0]
            w_eq, h_eq = 180, 44
            status = "in play"
            color_bg = (65, 72, 86)
            if m.phase is MatchPhase.ENDED:
                status, color_bg = "—", theme["spent"]
            elif m.phase is MatchPhase.PICK_ADVENTURER:
                status, color_bg = "between rounds", theme["idle"]
            elif m.phase is MatchPhase.DUNGEON:
                if eid in m.d_in_play:
                    status, color_bg = "ready", theme["active"]
                elif eid in sac_set:
                    status, color_bg = "sacrificed (bid)", theme["sac"]
                else:
                    status, color_bg = "used this run", theme["spent"]
            else:
                if eid in m.center_equipment:
                    status, color_bg = "on table", theme["active"]
                elif eid in sac_set:
                    status, color_bg = "sacrificed (bid)", theme["sac"]
                else:
                    status, color_bg = "—", theme["spent"]
            r = (ex, ey, w_eq, h_eq)
            pygame.draw.rect(surf, color_bg, r)  # type: ignore[union-attr]
            pygame.draw.rect(surf, theme["border"], r, 1)  # type: ignore[union-attr]
            t1 = name[:18]
            img1 = self._font_small.render(t1, True, _COLOR_TEXT)  # type: ignore[no-untyped-call, union-attr]
            img2 = self._font_small.render(status, True, _COLOR_DIM)  # type: ignore[no-untyped-call, union-attr]
            surf.blit(img1, (ex + 5, ey + 4))  # type: ignore[union-attr]
            surf.blit(img2, (ex + 5, ey + 24))  # type: ignore[union-attr]
            crossed = status.startswith("sacrific") or "used" in status
            if crossed and m.phase is not MatchPhase.PICK_ADVENTURER and m.phase is not MatchPhase.ENDED:
                ax1, ay1, ax2, ay2 = ex + 6, ey + 4, ex + w_eq - 6, ey + h_eq - 4
                for wLine in (4, 2):
                    pygame.draw.line(  # type: ignore[union-attr]
                        surf, (255, 70, 70), (ax1, ay1), (ax2, ay2), wLine
                    )
                    pygame.draw.line(  # type: ignore[union-attr]
                        surf, (255, 70, 70), (ax2, ay1), (ax1, ay2), wLine
                    )

        if a and a.kind in ("deck_to_hand", "hand_to_pile", "sacrifice") and t_anim < 0.99:
            self._draw_flying_card(surf, a, now)
        if a and a.kind == "dungeon_reveal" and t_anim < 0.99 and m.d_current:
            uu = self._lerp(t_anim) * (1.0 - self._lerp(t_anim))
            br = 4 + int(5 * 4.0 * uu)
            pygame.draw.rect(  # type: ignore[union-attr]
                surf,
                (200, 200, 120) if t_anim < 0.3 else (90, 110, 90),
                (stx - br, sty - br, stw + 2 * br, sth + 2 * br + 4),
                2,
            )

        if overlay:
            self._draw_label(
                surf, 20, self.config.height - 32, overlay, _COLOR_WARN
            )

        self._draw_run_outcome_banner(surf, now)
        pygame.display.flip()  # type: ignore[union-attr]

    def _rect_header(self, surf: object, x: int, y: int, text: str, w: int, h: int) -> None:
        pygame = self._pygame
        pygame.draw.rect(surf, _COLOR_PANEL, (x, y, w, h))  # type: ignore[union-attr]
        img = self._font.render(text, True, _COLOR_TEXT)  # type: ignore[union-attr, no-untyped-call]
        surf.blit(img, (x + 8, y + 6))  # type: ignore[union-attr]

    def _rect_titled(
        self,
        surf: object,
        r: tuple[int, int, int, int],
        text: str,
        border_rgb: tuple[int, int, int] = _COLOR_ACCENT,
    ) -> None:
        x, y, w, h = r
        lines = [ln for ln in str(text).split("\n") if ln][:2]
        pygame = self._pygame
        pygame.draw.rect(surf, _COLOR_CARD, (x, y, w, h))  # type: ignore[union-attr]
        pygame.draw.rect(surf, border_rgb, (x, y, w, h), 1)  # type: ignore[union-attr]
        yy = y + 6
        for line in lines:
            img = self._font_small.render(line[:20], True, _COLOR_TEXT)  # type: ignore[union-attr, no-untyped-call]
            surf.blit(img, (x + 4, yy))  # type: ignore[union-attr]
            yy += 18

    def _draw_label_lg(
        self, surf: object, x: int, y: int, text: str, color: tuple[int, int, int]
    ) -> int:
        img = self._font_lg.render(text, True, color)  # type: ignore[no-untyped-call, union-attr]
        surf.blit(img, (x, y))  # type: ignore[union-attr]
        return int(img.get_height())  # type: ignore[no-untyped-call, union-attr]

    def sync(self, m: "Match") -> None:
        if not self._open:
            return
        from dungeon_runner.match import MatchPhase

        self._table_top = self._y_table_top(m)
        old = self._last_m
        # Flash HP / monster
        now = int(self._pygame.time.get_ticks())  # type: ignore[union-attr]
        if m.phase is not MatchPhase.DUNGEON:
            self._prev_d_current_id = None
        prev_id = self._prev_d_current_id[0] if self._prev_d_current_id is not None else -3
        cur = m.d_current
        new_id = cur.def_id if cur is not None else -1
        old_hp = self._prev_hp
        if m.phase is MatchPhase.DUNGEON and cur is not None and new_id != prev_id:
            self._flash = _Flash(
                int(now + self.config.flash_duration_ms * 1.2), "monster"
            )
        if old_hp is not None and m.d_hp < old_hp and m.phase is MatchPhase.DUNGEON:
            self._flash = _Flash(
                int(now + self.config.flash_duration_ms * 0.9), "hp"
            )
        if cur is not None:
            self._prev_d_current_id = (cur.def_id, m.dungeon_sub.name if m.dungeon_sub else "")
        else:
            self._prev_d_current_id = None
        self._prev_hp = m.d_hp if m.phase is MatchPhase.DUNGEON else None

        if m.phase is MatchPhase.DUNGEON and (old is None or old.phase is not MatchPhase.DUNGEON):
            self._run_outcome = ""
            self._run_outcome_detail = ""
            self._run_outcome_until = 0
        if old and old.phase is MatchPhase.DUNGEON and m.phase is not MatchPhase.DUNGEON:
            for line in reversed(m.dungeon_run_log or []):
                n = (line or "").replace("\u2014", "-").replace("\u2013", "-")
                if "Result: Success" in n:
                    self._run_outcome = "success"
                    self._run_outcome_detail = n.strip()[:120]
                    self._run_outcome_until = now + int(self.config.run_outcome_banner_ms)
                    break
                if "Result: Failure" in n:
                    self._run_outcome = "failure"
                    self._run_outcome_detail = n.strip()[:120]
                    self._run_outcome_until = now + int(self.config.run_outcome_banner_ms)
                    break

        self._detect_animation(old, m)
        self._last_m = m
        self._draw(m)

    def _drain_one_animation(self) -> bool:
        import pygame  # type: ignore[import-untyped]

        a = self._active_anim
        m = self._last_m
        if not a or m is None:
            return self._open
        pg = self._pygame
        while self._open and a and pg.time.get_ticks() < a.t0 + a.duration:  # type: ignore[union-attr]
            for e in pg.event.get():  # type: ignore[union-attr]
                if e.type == pygame.QUIT:  # type: ignore[union-attr]
                    self.close()
                    return False
            self._draw(m, now_override=pg.time.get_ticks())  # type: ignore[union-attr]
            if self._clock is not None:
                self._clock.tick(60)  # type: ignore[union-attr, no-untyped-call]
        self._active_anim = None
        if self._last_m is not None:
            self._draw(self._last_m)
        return self._open

    def _pump_until(self, end_tick: int) -> bool:
        import pygame  # type: ignore[import-untyped]

        pg = self._pygame
        while self._open and pg.time.get_ticks() < end_tick:  # type: ignore[union-attr]
            for e in pg.event.get():  # type: ignore[union-attr]
                if e.type == pygame.QUIT:  # type: ignore[union-attr]
                    self.close()
                    return False
            if self._last_m is not None:
                self._draw(self._last_m)
            if self._clock is not None:
                self._clock.tick(60)  # type: ignore[union-attr, no-untyped-call]
        return self._open

    def pump(self, duration_ms: float) -> bool:
        if not self._open:
            return False
        import pygame  # type: ignore[import-untyped]

        pg = self._pygame
        while self._open and self._active_anim is not None:
            if not self._drain_one_animation():
                return False
        t0 = int(pg.time.get_ticks())  # type: ignore[union-attr]
        target = t0 + int(duration_ms)
        if self._run_outcome and self._run_outcome_until and self._run_outcome_until > t0:
            target = max(target, int(self._run_outcome_until))
        if self._last_m is not None and self._flash.until_tick and self._flash.until_tick > t0:
            target = max(target, int(self._flash.until_tick))
        if not self._pump_until(target):
            return False
        return self._open

    def show_static(
        self,
        m: "Match",
        message: str | None = None,
    ) -> bool:
        if not self._open:
            return False
        import pygame  # type: ignore[import-untyped]

        self._last_m = m
        self._draw(m, overlay=message)
        while self._open:
            for e in self._pygame.event.get():  # type: ignore[union-attr]
                if e.type == pygame.QUIT:  # type: ignore[union-attr]
                    self.close()
                    return False
                if e.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):  # type: ignore[union-attr]
                    return self._open
            if self._clock is not None:
                self._clock.tick(30)  # type: ignore[union-attr, no-untyped-call]
        return False
