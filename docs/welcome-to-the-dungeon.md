# Welcome to the Dungeon — design reference

Primary source: [IELLO English rulebook (PDF)](https://iellogames.com/wp-content/uploads/2016/08/Welcome-to-the-Dungeon-EN-Rulebook_web.pdf).

This file is the **living spec** for a simulator, written from a **verified physical copy**. Extend it for errata, variants, and implementation decisions.

---

## Goal

- Traverse **two** dungeons successfully → win (second **Success** card).
- Fail **two** dungeons → eliminated; alternatively win if **last player** remaining.

---

## Components

| Item | Count / notes |
|------|----------------|
| Monster cards | **13** → Monster Deck (deck list verified on physical copy) |
| Player aids | **4** (white / red = one failure) |
| Adventurer tiles | **4** (one shared hero per round; **not** equipment — see [Equipment](#equipment)) |
| Equipment tiles | **24** (six per adventurer; sacrificable in bidding — not the adventurer tile) |
| Success cards | **8** in box; **5** in play; **3** in box |
| Rule booklet | — |

---

## Setup

1. Choose or random **Adventurer**; place in center (first game suggests **Warrior**).
2. Place the **adventurer** tile and **six** matching **equipment** tiles in the center (names under [Equipment](#equipment)).
3. Shuffle **all** Monster cards → face-down **Monster Deck**.
4. Each player gets a **player aid**, **white** side up.
5. **Dungeon pile** — face-down stack for monsters added this round.
6. Set **5** Success cards nearby; leave **3** in the box.
7. Random **start player**.

---

## Round structure

### Bidding phase (clockwise)

On your turn, exactly one of:

- **Draw** the top of the Monster Deck, look **only yourself**, then either  
  1. **Add** the monster face-down to the **Dungeon pile**, or  
  2. **Place** the monster face-down **in front of you** and **sacrifice** one **equipment** tile from under the adventurer onto it (never the **adventurer** tile; monster + sacrificed **equipment** discarded for the **rest of the round**).  
     If the adventurer has **no** equipment left, you **must** add the monster to the dungeon.
- **Pass** — out until next round.

**Empty Monster Deck:** you **must pass** if you would draw.

**Public:** anyone may **count** Dungeon pile cards; may **not** look at faces.

**End of bid:** exactly **one** player has not passed → that player **runs the dungeon** (same adventurer tile + any **remaining** equipment).

### Dungeon phase (that player only)

1. **Starting HP** for the run = **adventurer tile** HP + sum of **remaining equipment** HP modifiers (per tile rules). The adventurer tile is never discarded.
2. **Before the first dungeon reveal:** if you still have **Vorpal Sword** or **Vorpal Dagger**, declare your **Vorpal blade** target for this run (see [Vorpal blade](#vorpal-blade-warrior-and-rogue)).
3. Reveal the Dungeon pile **one card at a time** (order matters). The runner learns the dungeon **only as each card is revealed**; unrevealed cards are **unknown** (no UI spoiler for the card under Polymorph—using it is a **gamble**).
4. For each revealed monster, resolve **eliminations** in a fixed priority (implementation must match the physical rulebook where it is precise): **(a)** **Vorpal blade** if this is the **first** copy of your named **species** this run (tile is **spent**); **(b)** other **legal** auto-defeats—where **only one** effect changes **observable** state, use the **narrower** rule first (e.g. **Ring of Power** before **Torch** on a goblin so **Ring** healing applies, because both would defeat the card but outcomes differ). **Holy Grail** and **Torch** are **not** removed when they banish a monster; if both could defeat the same card (e.g. even strength **and** strength **≤ 3**), **order does not matter** for table outcomes—either way the monster is gone and both tiles stay. **(c)** remaining **icons** / special text / **adventurer**; **(d)** otherwise lose HP equal to monster **strength**. Then **discard** the monster unless it **ended the run** by killing the hero (see **Monster cards** under [Equipment](#equipment)).
5. **Outcome**
   - **Success:** you **fully resolve** the **dungeon pile** in order; **after the last card**, **current HP** is **&gt; 0** → you **survived** → take a **Success** card (second success wins the game).
   - **Failure:** the only dungeon **fail** is **death**—HP **≤ 0** and not saved (e.g. **Healing Potion**). If the adventurer is **Mage**, **Omnipotence** is still an **active** equipment tile, and **[Omnipotence](#omnipotence-mage)** could apply, resolve it **before** finalizing defeat; it may turn the run into a **win**. If the run is still a loss, flip aid white→red, or if already red → **eliminated**; if one player remains → they win.
6. **Reveal** all monsters that were set aside with sacrifices this round (still happens after a **normal** success, after an **Omnipotence** success, or after a finalized **failure**—end-of-round cleanup).

### New round

- **Shuffle** all monster cards face-down.
- The player who **entered** picks the **next** Adventurer (Warrior / Barbarian / Mage / Rogue) and places that hero’s **adventurer** tile plus **six** equipment tiles in the center.
- **Start player** = that player (if eliminated → **left** neighbor).

---

## Monsters

Monster **strengths**, **deck counts**, and **neutralization icons** on cards are from a **physical copy** (13 cards total).

**Six canonical symbols** (for code / logs): **Torch**, **Chalice**, **Hammer**, **Cloak**, **Pact**, **Staff**. The same icon can map to **different equipment names** per adventurer — see each hero under [Equipment](#equipment).

| Monster | Strength | Count | Icons on card |
|---------|----------|-------|-----------------|
| Goblin | 1 | 2 | Torch |
| Skeleton | 2 | 2 | Torch, Chalice |
| Orc | 3 | 2 | Torch |
| Vampire | 4 | 2 | Chalice |
| Golem | 5 | 2 | Hammer |
| Lich | 6 | 1 | Chalice, Cloak |
| Demon | 7 | 1 | Pact, Cloak |
| Dragon | 9 | 1 | Staff, Cloak |
| **Total** | — | **13** | — |

---

## Equipment

**Source:** physical tiles (transcribed below).

### Adventurer tile (all heroes)

The **adventurer tile** is **not** an equipment tile. It **cannot** be discarded or sacrificed; it **represents** the adventurer **running the dungeon** in the dungeon phase. Only the **six equipment tiles** beneath it may be **sacrificed** in bidding and are subject to per-tile “used” rules.

### Monster cards (dungeon run)

During a dungeon, a monster either **kills** the hero (HP failure—run ends) or, once **fully resolved**, goes to the **discard** pile. There is **no** separate staging pile for “defeated,” “bypassed,” or polymorphed monsters—they follow the same **discard** rule unless the hero dies first.

### Single-use equipment (remove from play)

Some tiles say **once per dungeon**; others are **single-use** in the same **physical** way: when the effect is **spent**, **remove that equipment tile from play** for the rest of the dungeon (it cannot fire again this run). That includes **Vorpal Sword** / **Vorpal Dagger** ([Vorpal blade](#vorpal-blade-warrior-and-rogue)), **Healing Potion**, **Fire Axe**, **Polymorph**, **Demonic Pact** when its automatic defeats resolve, and any other tile your copy treats the same way.

### Base HP (adventurer tiles)

| Hero | HP |
|------|-----|
| Warrior | 3 |
| Barbarian | 4 |
| Rogue | 3 |
| Mage | 2 |

### Warrior (six equipment tiles)

| Tile | Effect | Icon |
|------|--------|------|
| Plate Armor | +5 HP | — |
| Knight Shield | +3 HP | — |
| Vorpal Sword | **Vorpal blade** — see [below](#vorpal-blade-warrior-and-rogue) | — |
| Torch | Defeat monsters with strength **≤ 3** | Torch |
| Holy Grail | Defeat monsters with **even** strength | Chalice |
| Dragon Spear | Defeat the **Dragon** | Staff |

No **Hammer**, **Cloak**, or **Pact** icon equipment; use another hero this round, **Vorpal**, or specials as needed.

### Barbarian (six equipment tiles)

| Tile | Effect | Icon |
|------|--------|------|
| Healing Potion | When HP would hit **0** or drop **below 0** for any reason: revive **once** per dungeon at **adventurer-tile HP only** (not equipment HP); **negates** that lethal outcome so the run can continue; then **remove from play** | — |
| Leather Shield | +3 HP | — |
| Chainmail | +4 HP | — |
| Fire Axe | Defeat **one** monster **after** it is **revealed** from the **dungeon pile**; **once** per dungeon (then **remove from play**) | — |
| Torch | Defeat monsters with strength **≤ 3** | Torch |
| War Hammer | Defeat **Golems** | Hammer |

No **Chalice**, **Cloak**, **Pact**, or **Staff** icon equipment.

### Mage (six equipment tiles)

| Tile | Effect | Icon |
|------|--------|------|
| Wall of Fire | +6 HP | — |
| Holy Grail | Defeat monsters with **even** strength | Chalice |
| Omnipotence | **Win the round** if every monster in the dungeon is a **different species**—see [timing](#omnipotence-mage) | — |
| Bracelet of Protection | +3 HP | — |
| Polymorph | **Dungeon phase only:** when you **reveal** a monster from the **dungeon pile** you cannot handle (e.g. a **Dragon**), **once** per dungeon you may replace **that** card with the **next unrevealed** card still **in the dungeon pile** (you resolve that card instead—maybe a **Goblin**). If there is **no** next card (**this** reveal was the **last** in the pile), **Polymorph** cannot be used—you cannot replace. On use, **remove from play** | — |
| Demonic Pact | **Automatically defeat** the **Demon** and **automatically defeat** the **next** monster in **reveal order**—**no** other defeat rules interact with those two cards; they are simply removed **without** HP loss. If the **Demon** is the **last** card in the pile, there is **no** next monster—the second clause does **nothing**. After that resolution, **remove from play** | Pact |

**Polymorph** is **not** used during bidding. Replacement comes from **deeper in the dungeon pile** only (not the **Monster Deck**). If this reveal is the **last** in the pile, Polymorph cannot be used. The **replaced** monster is **defeated** and **discarded** like any resolved monster that did not kill the hero ([Monster cards](#monster-cards-dungeon-run)).

### Omnipotence (Mage)

Only while the **Omnipotence** equipment tile is **still active** (not sacrificed in bidding).

You **do not** evaluate this during normal reveals—the **Mage** does **not** know if Omnipotence will save them **until** they would **die** (**HP ≤ 0** and not saved—**death** is the **only** dungeon fail condition).

**Then**, before locking in that defeat: gather **every monster** that belongs to the **entire dungeon** for this round—**seen** or **not yet seen**, **defeated** (already in **discard**) or **not yet resolved**—including the **dungeon pile**, monsters set aside with **sacrifices** (even if step **6** has not revealed them yet), and any other monster cards that were part of this run. Flip them **all face up** for the check. **Spoilers are fine here:** either Omnipotence is about to **flip the result to a win**, or the player **dies** anyway.

**Distinctness:** if **no** monster **species** appears **more than once** across that **full** set, **Omnipotence** triggers—the dungeon is a **win** (e.g. **Success** card). If any **duplicate** species exists, **death** stands. The **Mage** only learns whether Omnipotence saved them **after** “dying” the normal way.

**UI / agents:** do not resolve or leak Omnipotence **before** this moment; during normal play, keep hidden information hidden. The **full flip** is deliberate and only at this branch.

No **Torch**, **Hammer**, **Staff**, or **Cloak** icon equipment (besides **Chalice** and **Pact**-linked text above).

### Rogue (six equipment tiles)

| Tile | Effect | Icon |
|------|--------|------|
| Mithril Armor | +5 HP | — |
| Healing Potion | When HP would hit **0** or drop **below 0** for any reason: revive **once** per dungeon at **adventurer-tile HP only** (not equipment HP); **negates** that lethal outcome so the run can continue; then **remove from play** | — |
| Ring of Power | Defeat monsters with strength **≤ 2**; add their **total** strength to your HP (**healing**; current HP may exceed adventurer tile + equipment). **Narrower than Torch** (strength **≤ 3**): Rogue has **no** Torch tile, but if a house rule or variant ever gives both, **Ring** must win so the **healing** applies when both would defeat the same card | — |
| Buckler | +3 HP | — |
| Vorpal Dagger | **Vorpal blade** — same rules as **Vorpal Sword** ([below](#vorpal-blade-warrior-and-rogue)) | — |
| Invisibility Cloak | Defeat monsters with strength **≥ 6** | Cloak |

No **Torch**, **Chalice**, **Hammer**, **Pact**, or **Staff** icon equipment (use **Ring** / **Vorpal** / **Cloak** text and other heroes as needed).

### Vorpal blade (Warrior and Rogue)

**Equipment:** **Vorpal Sword** (Warrior) and **Vorpal Dagger** (Rogue). Same rules.

1. At the **very start** of the dungeon run, **before** the **first** card is flipped from the **dungeon pile**, declare **one monster species** (e.g. *Dragon*, *Skeleton*) that your vorpal blade will hunt this run.
2. Whenever that **species** is **revealed** from the dungeon pile, that card is **defeated automatically** (no HP loss), **but only for the first time** that species appears in **reveal order**. **Immediately after** that **first** auto-defeat resolves, **remove the Vorpal equipment tile from play** ([Single-use equipment](#single-use-equipment-remove-from-play)). Later copies of the **same** species in the same dungeon are **not** auto-defeated by vorpal—resolve them normally (other equipment, HP loss, etc.).
3. If your declared **species** never appears, the **Vorpal** tile was never **spent**—it stays in play until the dungeon ends (still unused).

If you do not have the vorpal equipment (sacrificed earlier or not part of this hero), skip the declaration.

---

## Implementation notes (pygame / RL)

### Information (bidding phase)

The game is **2–4 players**. Each seat may be a **human** or an **agent**, in any mix.

**What each seat knows about the dungeon pile:** they **never** inspect other players’ face-down adds. In code / UI, treat a seat as knowing **only** the **identities** of monsters **they personally added** to the pile this round; **others’** adds stay **unknown** (backs only, or no detail). This matches the table rule that anyone may **count** pile cards but may **not** look at faces ([Bidding phase](#bidding-phase-clockwise)); counting is **public** and belongs in every honest observation.

**Fairness vs agents:** an agent can remember its own adds perfectly; humans cannot. The **UI** should therefore list each player’s **known** contributions to the dungeon as a **memory aid** for humans—same information the agent already has in state.

Other observation choices (e.g. full cheat mode for debugging) can exist as flags; the default for play should match the above.

### Information (dungeon phase)

The **runner** (and any UI / agent tied to that seat) only learns dungeon contents **card by card** as the pile is **revealed**. Do **not** expose unrevealed faces or order beyond what the rules allow—**Polymorph** must stay a **gamble**. For **Mage** with **Omnipotence**, do **not** resolve or leak an Omnipotence “save” until the **post-death** full gather/flip ([Omnipotence](#omnipotence-mage)); that flip is allowed to **spoil** everything because the run has already **collapsed to death** or is about to be **retroactively saved**.

### Actions

Pass; draw then add **or** sacrifice + **which** equipment (when any equipment remains under the adventurer).

### Interface

**Text-first** client for this repo is enough (console / TUI / log-style). No requirement for polished graphics here.

### Effect order

Use [Dungeon phase](#dungeon-phase-that-player-only) step **4**. Prefer **narrow-before-wide** only when it changes what the players see (e.g. **Ring** healing vs **Torch** alone). Do not treat **Holy Grail** vs **Torch** as a player-facing priority contest—neither is consumed on use, and the banish result is the same if both apply.

### Success (dungeon)

**Win (usual):** after the **last** dungeon-pile card is **fully resolved**, **current HP &gt; 0**. **Alternate:** **[Omnipotence](#omnipotence-mage)** can grant a win **after** a normal loss is about to apply. Track **current HP** for combat and healing only.

### Long-term (outside this spec)

Repo-level roadmap (graphical **web** client later, browser-run trained model, etc.) lives in the **README**; this file stays the **rules** and **sim semantics** only.

### Repo integration (README)

**PettingZoo**, **TensorFlow** / training loop choices, and **observation / reward** layout are documented in the **README** so this spec does not drift into toolchain detail. **Turn shape:** bidding is **clockwise** one seat at a time; the **dungeon phase** is **runner-only** for choices ([Dungeon phase](#dungeon-phase-that-player-only))—any AEC wrapper must keep that split and must **not** imply every seat still takes dungeon turns like bidding.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-22 | Initial PDF extract |
| 2026-04-22 | Physical verification: deck, icons, four adventurers; doc tightened |
| 2026-04-22 | Adventurer tile vs equipment: definition, bidding/dungeon wording, six-tile tables |
| 2026-04-22 | Terminology: standardized on **equipment** (six tiles) vs adventurer tile |
| 2026-04-22 | Removed PDF-vs-box section; **Vorpal blade** timing (declare before first reveal; first instance only) |
| 2026-04-22 | **Polymorph** (Mage): dungeon-phase; swap with **next in dungeon pile**; unusable if none (e.g. last card) |
| 2026-04-22 | Designer **optional** (forced add on first bid) — **out of scope**, removed from spec |
| 2026-04-22 | Implementation notes: bidding partial info, human/agent seats, text UI; long-term → README |
| 2026-04-22 | Dungeon reveal-only info; effect precedence; Polymorph / Healing potion / Demon last (intermediate rules pass) |
| 2026-04-22 | Dungeon **success** = survive last card (**current HP &gt; 0**); removed HP-lost tally + FAQ |
| 2026-04-22 | **Omnipotence:** post-loss full flip; no reveal until normal fail; UI note |
| 2026-04-22 | Full-doc pass: Failure ↔ Omnipotence wording; step **6** after Omni win; Polymorph + duplicate check; residual gaps list |
| 2026-04-22 | Discard model; once/dungeon → remove tile; Fire Axe = reveal; Demonic Pact auto-defeats; Omni = entire dungeon; death only fail; Ring &gt; Torch |
| 2026-04-22 | **Single-use equipment** section; **Vorpal** + **Demonic Pact** remove from play when spent |
| 2026-04-22 | Bidding info wording + **Repo integration (README)**; public pile count; AEC vs runner-only dungeon |
| 2026-04-22 | Dungeon step **4**: Grail vs Torch order immaterial when both defeat; Ring-before-Torch only where outcomes differ |
| 2026-04-22 | Rogue **Ring** table: clarify “narrower than Torch” vs Rogue having no Torch tile |
