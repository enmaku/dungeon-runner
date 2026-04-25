"""RL reward scale constants (tunable)."""

# Match end: second success (two victory cards) vs last one standing
REWARD_MATCH_WIN_SUCCESS = 10.0
REWARD_MATCH_WIN_STANDING = 4.0
REWARD_MATCH_LOSE = -3.0

# Dense: agent who just acted (kept small relative to match/dungeon terms)
REWARD_PER_ACTION = 0.00025

# Dungeon (runner)
REWARD_DUNGEON_SUCCESS = 1.5
REWARD_DUNGEON_FAIL = -0.45

# Runner eliminated / aid to red in fail path is reflected via dungeon fail and later match
