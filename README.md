# dungeon-runner

Train a neural network to play [*Welcome to the Dungeon*](https://iellogames.com/games/welcome-to-the-dungeon/) (IELLO / Oink Games): a pygame-backed game environment with **TensorFlow** (`tf.keras`) for the policy and **TensorBoard** for run metrics—**custom** training loops in Python (no general-purpose RL framework as the primary path).

## Docs

- [`docs/welcome-to-the-dungeon.md`](docs/welcome-to-the-dungeon.md) — **rules and sim semantics** from a verified physical copy (monsters, equipment, round flow, implementation notes for parity/fairness). This file does **not** duplicate toolchain choices here; the README is the source for **PettingZoo**, **TensorFlow**, and RL-facing observation/reward design.

## Scope (near term)

- **Players:** **2–4** seats; each seat is either a **human** or an **agent**
- **Human at the table:** implementations can assume **one human** + **N** agents for a long time; **multi-human** seating and input routing are **out of scope** until you explicitly need them.
- **UI:** **text-first** is fine (console / TUI / structured logs). No dependency on a polished graphical client in this repo.
- **Partial observability:** per the rules doc, seats don’t see others’ dungeon adds; **what each seat is allowed to know** in code is spelled under **Observation / state** below. **Human UI:** list each seat’s **own** dungeon adds as a memory aid so humans match agent state ([Information (bidding phase)](docs/welcome-to-the-dungeon.md#information-bidding-phase) in the rules doc).

## Architecture (decided)

- **Environment API:** [**PettingZoo**](https://pettingzoo.farama.org/) **AEC** (agent–environment–cycle)—**bidding** follows **clockwise** seat order with one `agent_selection` at a time. **Dungeon phase** is **runner-only** for meaningful actions and observations ([rules doc](docs/welcome-to-the-dungeon.md#dungeon-phase-that-player-only)); other seats are idle until the next round—model the env that way (e.g. only the runner steps, or others get masked / no-op transitions), not as if every seat still bid in circle during the dungeon.
- **Training:** the game has **no solo mode**; every training episode is **multi-seat** in one match (typically **one shared policy** stepping for whichever seat is active—see **Policy / weights**). **Player count:** sample **2, 3, or 4** seats **at random every episode** (2p and 4p play very differently—you want the policy exposed to all). **No** staged “ramp” difficulty—the game does not offer a clean curriculum axis beyond seat count, so we do **not** plan one.
- **Policy / weights (v1):** **one** `tf.keras` model with **shared trainable weights** across all seat indices—every seat’s updates apply to the **same** parameters (you may still feed **seat index** or symmetry-breaking features in the observation if useful). **Personalities** stay in **wrappers** (temperature on logits, priors, ε-biases, etc.). **Separate weight copies** per seat remain a later ablation if you want them; switching is mostly “instantiate N models vs 1” plus how you aggregate gradients, not a rules-engine change.
- **Playtesting:** you sit in **one** seat; any other seats can be agents (any mix of wrappers / checkpoints).

### Observation / state (decided)

**Bidding — current seat always includes:**

- **Dungeon pile size** (card count)—**public** to all seats per rules ([public count](docs/welcome-to-the-dungeon.md#bidding-phase-clockwise)); include it in every honest observation.
- **Their own adds** to the dungeon (full card identities), not other seats’ hidden adds.
- **Equipment still available** — the **six equipment tiles** (minus sacrifices this round) **under the shared adventurer** in the center, per [Equipment](docs/welcome-to-the-dungeon.md#equipment) / bidding text—not a separate informal “pool.”
- If applicable: **the card they drew** from the pile (after a draw), until the rules clear or replace that knowledge.

**Dungeon / choice points:**

- **Who acts:** only the **runner** (the single seat who won the bid) takes dungeon actions; the rules doc is explicit ([Dungeon phase](docs/welcome-to-the-dungeon.md#dungeon-phase-that-player-only)).
- The runner uses the **same information they had at bidding** (no resetting memory between phases).
- **Plus** everything they have **already revealed** from the **dungeon pile** this run (order and identities matter where the rules care—e.g. **Polymorph**, **Fire Axe**).

**Design stance:**

- You expect **current-state–only** to be enough for learning in this small game—no play-history channel in v1 (who passed when, who removed which equipment, etc.). Those can be added later if training stalls.
- **Vectorization:** **fairly maximal** for v1: encode a **full, legal, current snapshot** the seat may use to decide (fixed layouts, embeddings for IDs, legal-action masks as needed)—not a stripped-down minimal encoding. Trim history, not present-state richness.

### Learning (decided)

- **Goal:** this project is partly a **learning exercise**—you want to **implement the RL update yourself** (policy gradients / advantages / whatever you choose), not hide that work behind a general-purpose RL framework for v1.
- **Stack:** **TensorFlow**, using the **Keras** API (`tf.keras`) for the **network** (layers, forward pass, etc.). Training logic (sampling actions, computing losses, applying gradients) lives in **your own Python**, calling into Keras models and TF optimizers.
- **Out of scope for now:** adopting **Ray RLlib**, **TF-Agents**, etc. as the *primary* training path. Revisit only if you deliberately choose to after the custom loop exists.

### Rewards (decided — high level)

- **Largest:** winning the **overall match** (second **Success** card, or last player standing after others are eliminated).
- **Large (smaller than match win):** **surviving a dungeon** successfully (getting a **Success** toward the match—includes **Omnipotence** saves, since that still counts as a dungeon win).
- **Severe penalty:** **dying** in the sense that ends or derails you badly—treat as **elimination** (failed two dungeons / aid goes red twice) **and/or** a **dungeon run that ends in death** without a save, depending on what you wire; the point is “**being out**” or “**throwing the run**” should hurt a lot more than missing a marginal bid.

Exact numeric scales stay **TBD** in code; philosophy is **strong match signal**, **strong dungeon signal**, **heavy stick** for terminal failure.

**Edge cases / wrong lessons to watch for**

- **Omnipotence:** the runner **hits** the death branch, then **wins** the dungeon anyway. They must get the **dungeon success** reward (and **not** the full “you died” penalty), or the net teaches “avoid Omnipotence lines.” Spell this in the env’s reward hook when you implement it.
- **Win without dungeon heroics:** winning because **everyone else was eliminated** is still a **match** win—credit that; don’t accidentally only reward “I entered every dungeon.”
- **Who gets the number (wire-time choice):** when you implement PettingZoo returns, decide whether **only** the **active** agent gets step rewards, or everyone gets **terminal** payouts from their own seat’s perspective—and document the choice next to the reward hook.
- **Sparse mid-game:** long bidding with **no** dungeon yet means **long stretches with no learning signal** except the eventual big chunks—that’s OK for this design, but learning may be slow early; wrappers / self-play diversity help.
- **Good sacrifice looks dumb locally:** throwing away a torch to dodge a bad dungeon can look like “lost value” without match-level context; your scheme avoids tiny per-action rewards, which is good—just don’t add small shaped rewards later that punish sacrifices by accident.

## Planned stack

| Piece | Role |
|--------|------|
| **Python** | Project language |
| **PettingZoo** | AEC multi-agent env interface (`parallel` not the default fit) |
| **pygame** | Timestep / display loop (can run headless for training) |
| **TensorFlow + Keras** | Neural net (`tf.keras`); **custom** training loop (no RL framework as primary path) |

Exact layout (`src/`, `tests/`, entrypoints) will appear once the simulator exists.

### Tooling (decided)

- **Python:** default to **current stable** from [python.org](https://www.python.org/downloads/) (bump the minor you develop on as new releases ship; only pin an older version if a dependency like TensorFlow lags).
- **Dependencies:** **`pip`** + **`requirements.txt`**—no Poetry / `uv` project as the default layout for v1.
- **CI:** **ad-hoc** until you explicitly want a pipeline; nothing committed here yet.

### Logging & checkpoints (decided)

- **TensorBoard** for scalars (loss, returns, win rate, etc.) and any extra summaries you want; standard **`tensorboard --logdir=...`** workflow.
- **Weights on disk** via **TensorFlow’s** checkpointing (`tf.train.Checkpoint` and/or `Model.save_weights`), organized under per-run directories alongside the TB log (no W&B / external experiment service as the default for v1).

## Quickstart

1. Create a virtualenv, then: `pip install -r requirements.txt` (or `pip install -e .`) and `pytest` to verify the game engine.
2. **PPO training (optional):** `pip install -e ".[train]"`, then e.g. `python scripts/train.py --logdir runs/my_run` (long run: default is 10k PPO updates; add `--updates 5` for a quick smoke test). Weights land at `logdir/policy.weights.h5` (also rewritten every `--save-every` steps, default 500, while training); run `tensorboard --logdir=runs/my_run` for scalars (PPO `loss/*`, `rollout/*`, and `game/*` such as `nn_win_rate` and self-play vs mixed-bot fraction). Default is CPU; for GPU, install a CUDA-enabled TensorFlow build and use your usual device env vars.
3. **Pygame table-style UI** (optional, for `scripts/play_random_game.py` with `--gui`): `pip install -e ".[gui]"`. Layout uses facedown cards and a full equipment row (X for sacrificed/used). Pass `--god` to show exact deck and draw faces; defaults are slow on purpose—use `--step-ms` / `--dungeon-step-ms` to tune.

## Long-term (not committed work)

These depend on how training goes and are **not** required for the current codebase:

- A **graphical** client, likely **web**-based, as a **separate project** if you pursue it.
- Running a **trained network in the browser** only really pays off if learning produces a strong policy worth shipping.

## Rulebook

Official English PDF from the publisher: [Welcome to the Dungeon — English rulebook](https://iellogames.com/wp-content/uploads/2016/08/Welcome-to-the-Dungeon-EN-Rulebook_web.pdf).

## License

Game rules and trademarks belong to their respective owners; this repository is only an independent implementation and research code. I'm doing this for my own education and because I love your game, please don't sue me IELLO.
