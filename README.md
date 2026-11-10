# Snake RL

Hand-rolled PyTorch DQN agent that learns to play Snake in Pygame — with checkpointing, resume, and independent evaluation. No gym or stable-baselines — just pygame, torch and numpy.

## Results

- **Best training score: 64** (improved agent, game 0224, 3 Nov 2026)
- **Greedy evaluation: averages 46** (5-episode check: average 46, median 45, best 52, worst 43, average 586.8 steps per game)

Evaluation is greedy with no exploration, so it usually scores higher than training.

## Setup

Python 3.14.

```bash
git clone https://github.com/Crazyreyes/snake_RL.git
cd snake_RL
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

`requirements.txt` pins `pygame-ce==2.5.8`, `torch==2.14.0` and `numpy==2.5.3`.

For CPU-only torch, install torch first, then the rest:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Play

Run from the repo root. Arrow keys or WASD to move, any key to play again after game over.

```bash
python snake.py
python snake_improved.py
```

- `snake.py` — the original Snake game (600×600, simple rectangles).
- `snake_improved.py` — the improved Snake game (680×680, grid, sprite head / body / tail and apple from `improved_assets/`).

## Train

Run from the `RL/` directory — the training scripts import `common` and `experiments` as top-level packages.

```bash
cd RL
python snake_improved_RL.py
```

Other trainers:

```bash
python snake_standard_RL.py
python snake_improved_double_RL.py
```

All three use epsilon 80 → 5 (decay 0.5), save a training checkpoint every 25 games, and save a new best model whenever a game beats the previous best score. Replays are saved for games scoring 50 or more (40 or more for Double DQN).

Resume training from the latest checkpoint:

```bash
python snake_improved_RL.py --resume
python snake_improved_RL.py --resume --checkpoint-every 25
```

Train headless, without opening a game window (faster, and no display needed):

```bash
python snake_improved_RL.py --no-render
```

`SNAKE_RL_RENDER=0` does the same thing without the flag. Close the game window to stop training when rendering is on. Training outputs are generated locally and are not committed:

- `RL/checkpoints/` — `best_model.pth` and `latest_training_checkpoint.pth` per experiment
- `RL/logs/` — per-game training CSV
- `RL/replays/` — saved high-scoring game replays
- `RL/evaluations/` — evaluation results

## Evaluate

Run from the `RL/` directory. This loads the best model for an experiment and plays it greedily, with no exploration and no learning.

```bash
cd RL
python -m experiments.evaluate --episodes 5
```

Defaults are `--experiment improved --episodes 100 --seed-start 10000`. To evaluate a different experiment or a specific checkpoint:

```bash
python -m experiments.evaluate --experiment standard --episodes 20
python -m experiments.evaluate --checkpoint RL/checkpoints/snake_improved_RL/best_model.pth --episodes 20
```

Results are printed and saved as CSV + JSON in `RL/evaluations/`.

## Baseline: train and evaluate all three agents

Run all three trainers for the same number of games, headless, one after another, then evaluate each one — no need to start each script by hand:

```bash
cd RL
python run_baseline.py
```

That trains standard, improved and Double DQN for 100 games each, then evaluates each on 100 episodes. Other options:

```bash
python run_baseline.py --games 50 --episodes 20
python run_baseline.py --games 100 --skip-eval
python run_baseline.py --games 100 --render
```

Each trainer also accepts `--max-games` on its own, so it stops by itself instead of running until you close the window:

```bash
python snake_improved_RL.py --no-render --max-games 100
```

## Project structure

```
snake.py                  Play the original Snake game
snake_improved.py         Play the improved Snake game
assets/                   Sounds for the original game
improved_assets/          Sprites for the improved game (head, body, tail, apple)
RL/
  snake_improved_RL.py        Train the improved DQN agent
  snake_standard_RL.py        Train the standard DQN agent
  snake_improved_double_RL.py Train the improved Double DQN agent
  agents/dqn_agent.py         DQN agent
  common/
    config.py                 Grid, window, learning rate, render and paths
    snake_game.py             Snake game environment used for training
    checkpoint.py             Best-model and training-checkpoint save / load
    replay.py                 Replay recording
    renderer.py               Training renderer
    logger.py                 Training logging
    rewards.py                Standard and improved reward functions
  experiments/
    train_common.py           Shared training loop
    evaluate.py               Independent greedy evaluation
```

## License

MIT — see [LICENSE](LICENSE). The license covers this code only, not the sprites / assets.
