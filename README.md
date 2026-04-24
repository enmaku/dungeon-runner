# dungeon-runner

A Python implementation of [*Welcome to the Dungeon*](https://iellogames.com/games/welcome-to-the-dungeon/) (IELLO / Oink Games) with a **match engine**, a [**PettingZoo**](https://pettingzoo.farama.org/) **AEC** environment, optional **TensorFlow / Keras** PPO training (hand-written update loop, not a third-party RL library as the main path), optional **pygame** UI, and **TensorBoard** run metrics.

## What’s in the tree

| Area | Location |
|------|----------|
| Game rules and state | `src/dungeon_runner/match.py`, `actions.py`, `catalog.py`, … |
| PettingZoo env | `src/dungeon_runner/pettingzoo_aec.py` — `WtdAECEnv` (2–4 players) |
| RL: observations, action codec, PPO, model | `src/dungeon_runner/rl/` |
| Bots | `src/dungeon_runner/bots/` (e.g. weighted-random) |
| Table UI | `src/dungeon_runner/ui/pygame_view.py` (used by the play script, not by training) |
| Scripts | `scripts/train.py` (PPO vs bot), `scripts/train_rllib.py` (PPO, Ray-parallel self-play, same Keras loop), `scripts/play_random_game.py` |
| Rules reference (physical-game parity) | [`docs/welcome-to-the-dungeon.md`](docs/welcome-to-the-dungeon.md) |

Tests live under `tests/`. The package is installable with **`pip install -e .`** ([`pyproject.toml`](pyproject.toml)); optional groups are `dev` (pytest), `gui` (pygame), `train` (TensorFlow, PettingZoo, Ray, etc.). A thin [`requirements.txt`](requirements.txt) installs the editable package plus pytest for a minimal dev setup.

## Rules and behavior

Tournament rules, equipment, and phase flow that matter for the simulator (including what information each seat is allowed) are documented in the rules file above. The README only summarizes how this repo encodes that.

- **Bidding** runs in clockwise seat order with a single `agent_selection` in AEC. **Dungeon** actions are for the **runner** only; other agents do not act (see the rules link for the actual game). Training samples **2, 3, or 4** players per episode (`sample_episode` in `scripts/train.py`, `sample_episode_config` in `src/dungeon_runner/rl/rllib_keras_module.py`) so the policy sees varied table sizes.
- **Partial observability:** the observation builder follows “honest” information per seat (e.g. your own dungeon adds, public pile count, not others’ hidden cards). The vector is fixed-size: **`observation.OBS_DIM`** (87 floats) with an action mask, described in `src/dungeon_runner/rl/observation.py`. It is **current state only**—no opponent history channel.
- The pygame client lists each player’s own dungeon adds to align human play with the vector state (see the rules doc section on bidding information).

## Training

- **`PolicyValueModel`** in `src/dungeon_runner/rl/model.py` is a single shared-weights policy–value network used for every active seat. Hyperparameters like hidden width use **`DEFAULT_PPO_HIDDEN`** in that file.
- **`scripts/train.py`** runs a custom **PPO** loop: collects rollouts in-process against a **random** bot, logs TensorBoard scalars under `logdir/scalars/`, and saves **`logdir/policy.weights.h5`**. If `--weights` is omitted, it loads `logdir/policy.weights.h5` when that file exists.
- **`scripts/train_rllib.py`** runs the **same** Keras PPO update but uses **Ray** to collect self-play rollouts in parallel. It is **not** RLlib’s PPO `Algorithm`—Ray is for **sampling** only, because Ray 2.5+ does not match this project’s portable `Model.save_weights` / `load_weights` workflow. On **macOS**, if workers spin up poorly, use Ray’s [local resource notes](https://docs.ray.io/en/latest/ray-core/configure.html#local-cluster-setup).
- **Optimizer state is not saved**; switching between scripts or restarts you still load the same H5 **weights** but a fresh **Adam** state.

Reward scales for match and dungeon are centralized in `src/dungeon_runner/rl/rewards.py` and applied in `WtdAECEnv.step` (e.g. match end and dungeon completion or failure, including the Omnipotence-style dungeon success case when the run still counts as a win). TensorBoard tags include `loss/*`, `rollout/*` (e.g. `mean_reward`, `nn_transitions`), and `game/*` (e.g. win rate, episode length, truncation rate).

**Smoke:** after `pip install -e ".[train]"`, `python scripts/train.py --logdir runs/smoke --updates 5` and open TensorBoard on `runs/smoke/scalars` if you want a short trace.

## Pygame (optional)

`pip install -e ".[gui]"` and run `play_random_game.py` with `--gui` for a table layout (facedown cards, equipment row, `--god` to reveal the deck, `--step-ms` / `--dungeon-step-ms` to slow automation).

## Out of scope here

- Multi-seated **human** hot-seat routing: the UI and scripts assume a simple human + bots or full-bot setup.
- **CI** and external experiment services are not set up; logs and weights are local and TensorBoard-only by default.
- A separate **web** or browser client, if you ever want one, would be another project.

## Official rulebook

Publisher PDF: [Welcome to the Dungeon — English rulebook](https://iellogames.com/wp-content/uploads/2016/08/Welcome-to-the-Dungeon-EN-Rulebook_web.pdf).

## License

Game rules and trademarks belong to their respective owners; this repository is only an independent implementation and research code. I'm doing this for my own education and because I love your game, please don't sue me IELLO.
