"""RL reward scale constants (tunable)."""

# Match end: second success (two victory cards) vs last one standing
REWARD_MATCH_WIN_SUCCESS = 15.0
REWARD_MATCH_WIN_STANDING = 4.0
REWARD_MATCH_LOSE = -4.0

# Dense: agent who just acted (kept small relative to match/dungeon terms)
REWARD_PER_ACTION = 0.025

# Dungeon (runner): success scales with pile size + bidding sacrifices (equipment tiles removed
# during bidding only, not dungeon-phase use). Stored on Match as dungeon_run_reward_difficulty.
REWARD_DUNGEON_SUCCESS_PER_CARD = 0.35
REWARD_DUNGEON_FAIL = -0.45


def dungeon_success_reward(difficulty_units: int) -> float:
    n = max(1, int(difficulty_units))
    return REWARD_DUNGEON_SUCCESS_PER_CARD * float(n)


# Runner eliminated / aid to red in fail path is reflected via dungeon fail and later match
