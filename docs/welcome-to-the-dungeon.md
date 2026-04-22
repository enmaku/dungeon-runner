# Welcome to the Dungeon — design reference

Primary source: [IELLO English rulebook (PDF)](https://iellogames.com/wp-content/uploads/2016/08/Welcome-to-the-Dungeon-EN-Rulebook_web.pdf).  
This file is the project’s living spec: extend it whenever physical components or house rules supply details the PDF omits or garbles.

---

## Goal

- Traverse **two** dungeons successfully → win (second **Success** card).
- Fail **two** dungeons → eliminated; alternatively win if **last player** remaining.

---

## Components (from rulebook)

| Item | Count / notes |
|------|----------------|
| Monster cards | **13** (shuffle all into Monster Deck) |
| Player aids | **4** (white side / red side = 1 failure) |
| Adventurer tiles | **4** (shared hero each round) |
| Equipment tiles | **24** (six per adventurer) |
| Success cards | **8** in box; **5** used in play, **3** left in box |
| Rule booklet | — |

---

## Setup (summary)

1. Choose or random **Adventurer**; place center (first game suggests **Warrior**).
2. Line up **six** matching **Equipment** tiles under that adventurer (rulebook example: Warrior → Plate Armor, Knight Shield, Vorpal Sword, Dragon Spear, Holy Grail, Torch).
3. Shuffle **all** Monster cards → face-down **Monster Deck**.
4. Each player gets a **player aid**, **white** side up.
5. **Dungeon pile** (where monsters for this round go, face-down).
6. **5** Success cards at hand; other **3** stay in box.
7. Random **start player**.

---

## Round structure

### Bidding phase (clockwise)

On your turn, exactly one of:

- **Draw** top of Monster Deck, look **only yourself**, then:
  1. **Add** monster face-down to **Dungeon pile**, or  
  2. **Place** monster face-down **in front of you** and **sacrifice** one **Equipment** tile from under the adventurer onto that monster (both discarded for the **rest of the round**).  
     If the adventurer has **no** equipment left, you **must** add the monster to the dungeon.
- **Pass** — out until next round; no further bids.

**Empty Monster Deck:** if you would draw, you **must pass**.

**Public info:** anyone may **count** cards in the Dungeon pile; may **not** inspect faces.

**End of bid:** when **one** player has not passed, that player **enters the dungeon** alone with remaining equipment.

### Dungeon phase (dungeon runner only)

1. **Total HP** = adventurer HP + HP from **remaining** equipment (per tile rules).
2. Reveal Dungeon pile **one card at a time** (order matters for equipment).
3. For each monster: if you still have equipment that **eliminates** that monster, discard monster **without** losing HP; else lose HP equal to monster **strength**, then discard monster.
4. After all monsters resolved:
   - **HP lost** &lt; **HP you entered with** → success → take a **Success** card; **two** successes → win.
   - **HP lost** ≥ **HP entered** → failure → flip aid **white → red**, or if already red → **eliminated**. If one player left → they win.
5. **Reveal** all monsters that were discarded (sacrifice pairs) for the round.

### New round

- **Shuffle** all monster cards face-down.
- Player who **entered** the dungeon picks **next** Adventurer (Warrior / Barbarian / Mage / Rogue) and corresponding equipment in center.
- **Start player** = that player (if eliminated → player to their **left**).

---

## Monsters (rulebook “List of monsters”)

Strength values are authoritative in the rules; **exact deck counts per type are not** in the booklet (only **13** cards total). Fill the table after a physical inventory.

| Monster (verify names on cards) | Strength |
|----------------------------------|----------|
| Goblin | 1 |
| Skeleton | 2 |
| Orc | 3 |
| Vampire | 4 |
| Golem | 5 |
| Lich | 6 |
| Demon | 7 |
| Dragon | 9 |

**Deck composition (TBD — verify from box):**  
*(e.g. 2× Skeleton … — add rows/columns when confirmed.)*

---

## Equipment (partial — rulebook gaps)

### Warrior example (setup in rulebook)

Plate Armor, Knight Shield, Vorpal Sword, Dragon Spear, Holy Grail, Torch.

### Other adventurers (TBD)

For each of **Barbarian**, **Mage**, **Rogue**: list **six** equipment names and full effect text from tiles.

### Adventurer HP (partially in rules)

- **Healing Potion** example: **Barbarian** base HP **4**, **Rogue** base HP **3** (reset when potion used at ≤0 HP during draw resolution context in rules).
- **Warrior**, **Mage**: confirm from tiles.

### Tile clarifications (from rulebook; verify wording on components)

Summaries below are for implementation planning — **copy exact text from physical tiles** into a subsection later.

- **Ring of Power** — Example: Goblin + Skeleton in dungeon → defeat both and **add** their combined strength **to** your HP (unusual; confirm exact tile text).
- **Vorpal Sword / vorpal dagger** — Choose monster type **before** first dungeon reveal; defeats **all** monsters of that type in the dungeon (example mentions **two** Skeletons).
- **Vorpal Axe** — Choose monster to defeat **as soon as** you see the **current** card, before seeing **later** cards.
- **Omnipotence** — After revealing **all** dungeon cards: if **all monsters were different**, you **win the round** even if you would have failed.
- **Demonic Pact** — If **Demon** is the **last** monster in the dungeon, simply defeat it; second part of effect irrelevant then.
- **Polymorph** — If **Monster Deck** is **empty**, cannot use (timing: confirm on tile).
- **Healing Potion** — After drawing a monster card, if you have **0 or fewer** HP, may set HP to adventurer tile base (example: Barbarian 4, Rogue 3) and continue; confirm full timing from tile.

**Equipment not detailed in “Tile clarifications” (TBD):**  
Plate Armor, Knight Shield, Dragon Spear, Holy Grail, Torch, and **all** other-hero equipment — transcribe effects from copy.

---

## Optional variant (designer)

First turn of **bidding**: card you draw **must** be added to the dungeon (implement as config flag).

---

## Implementation notes (for pygame / RL)

- **Hidden information:** drawn-but-not-yet-placed monster is private; dungeon stack is hidden but **count** is public — define observation space deliberately (full cheat vs human-like).
- **Action space:** pass; draw then (add to dungeon **or** sacrifice + **which** equipment) when equipment exists.
- **Effect stack:** rulebook does not define a global resolution order when multiple items could apply; pick a **deterministic order** and document it here once chosen.

---

## Changelog

| Date | Change |
|------|--------|
| *(add rows)* | Initial extract from PDF + known gaps |

When you answer open questions (deck list, full equipment, HP, aids text), append sections or edit **TBD** blocks in place—no separate tool required.
