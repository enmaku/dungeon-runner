"""RL reward scale constants (tunable)."""

# Match end: second success (two victory cards) vs last one standing
REWARD_MATCH_WIN_SUCCESS = 1.0
REWARD_MATCH_WIN_STANDING = 1.0
REWARD_MATCH_LOSE = -1.0

# Dense: agent who just acted (kept small relative to match/dungeon terms)
REWARD_PER_ACTION = 0

# Dungeon (runner): success scales with pile size and bidding sacrifices (equipment tiles removed
# during bidding only, not dungeon-phase use). Counts stored on Match at dungeon start.
REWARD_DUNGEON_SUCCESS_PER_CARD = 0
REWARD_DUNGEON_SUCCESS_PER_TILE = 0
# Bidding: per equipment tile removed to stay in (SacrificeEquipment). Default ≈ half the eventual
# REWARD_DUNGEON_SUCCESS_PER_TILE credit if that run succeeds — discourages casual discards, still
# net-positive for a runner who clears the dungeon.
REWARD_BIDDING_TILE_DISCARD = 0
REWARD_BIDDING_PASS = 0
REWARD_DUNGEON_FAIL = 0


def dungeon_success_reward(n_cards: int, n_tiles: int) -> float:
    return 3.0
    # cards = max(1, int(n_cards))
    # tiles = max(0, int(n_tiles))
    # return REWARD_DUNGEON_SUCCESS_PER_CARD * float(cards) + REWARD_DUNGEON_SUCCESS_PER_TILE * float(tiles)


# Runner eliminated / aid to red in fail path is reflected via dungeon fail and later match
