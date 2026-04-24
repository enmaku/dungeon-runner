"""RL reward scale constants (tunable)."""

# Terminal
REWARD_MATCH_WIN = 5.0
REWARD_MATCH_LOSE = -3.0

# Dungeon (runner)
REWARD_DUNGEON_SUCCESS = 1.5
REWARD_DUNGEON_FAIL = -1.0

# Runner eliminated / aid to red in fail path is reflected via dungeon fail and later match
