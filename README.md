# dungeon-runner

Train a neural network to play [*Welcome to the Dungeon*](https://iellogames.com/games/welcome-to-the-dungeon/) (IELLO / Oink Games): a pygame implementation of the game as the environment, with TensorFlow (or related tooling) for learning.

## Docs

- [`docs/welcome-to-the-dungeon.md`](docs/welcome-to-the-dungeon.md) — rules summary, open questions, and **TBD** fields filled in as the box is verified.

## Planned stack

| Piece | Role |
|--------|------|
| **Python** | Project language |
| **pygame** | Human-readable (or headless) game loop and rendering |
| **TensorFlow** | Models, training, checkpoints |

Exact layout (`src/`, `tests/`, entrypoints) will appear once the simulator exists.

## Quickstart

*(To be written after `pyproject.toml` / `requirements.txt` and a runnable entrypoint exist.)*

Typical flow will be something like: create a virtualenv, install dependencies, run the game client or training script.

## Rulebook

Official English PDF from the publisher: [Welcome to the Dungeon — English rulebook](https://iellogames.com/wp-content/uploads/2016/08/Welcome-to-the-Dungeon-EN-Rulebook_web.pdf).

## License

Game rules and trademarks belong to their respective owners; this repository is only an independent implementation and research code. Add a `LICENSE` file for your own code when you choose one.
