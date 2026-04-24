from __future__ import annotations

import numpy as np

from dungeon_runner.match import MatchPhase
from dungeon_runner.pettingzoo_aec import WtdAECEnv
from dungeon_runner.types_core import AdventurerKind


def test_aec_reset_and_few_steps() -> None:
    e = WtdAECEnv()
    e.reset(seed=0, options={"n_players": 2, "start_seat": 0, "first_hero": AdventurerKind.WARRIOR})
    assert e.agent_selection in e.agents
    o = e.observe(e.agent_selection)
    assert o is not None
    assert o["obs"].shape[0] > 0
    n = 0
    for _ in range(50):
        o = e.observe(e.agent_selection)
        assert o is not None
        msk = o["action_mask"]
        if float(msk.sum()) < 0.5:
            break
        a = int(np.flatnonzero(msk > 0.5)[0])
        e.step(int(a))
        n += 1
        m = e._m  # noqa: SLF001
        if m is not None and m.phase is MatchPhase.ENDED:  # type: ignore[union-attr, misc, unused-ignore, SIM201]
            break
    assert n > 0
