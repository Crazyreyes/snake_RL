# Experiments

Both experiments train the same three agents (standard, improved,
improved Double DQN) on the same game, rewards, network
(15 → 128 → 3), epsilon (80 → 5, decay 0.5), learning rate (0.001),
gamma (0.9), memory (5000) and batch size (128). Only the loss
differs.

## Shared code (same for both experiments)

- `RL/run_baseline.py` — trains all three agents, then evaluates each
- `RL/common/renderer.py` and `--no-render` / `SNAKE_RL_RENDER=0` — headless training
- `--max-games` — stop automatically after N games
- `RL/experiments/train_common.py` — shared training loop
- `RL/experiments/evaluate.py` — independent greedy evaluation

## Experiment 1 — MSE loss

- Loss: `nn.MSELoss()`, no gradient clipping
- `train_long()` was originally a per-sample Python loop; the
  vectorised version in `RL/agents/dqn_agent.py` computes the same
  targets (verified: max difference 0.00e+00 on the same batch), so
  it does not change the learning, only the speed.

Run it:

```bash
cd RL
python snake_improved_RL.py --no-render --max-games 100 --loss mse
python run_baseline.py --games 100 --loss mse
```

## Experiment 2 — Huber loss

- Loss: `nn.SmoothL1Loss()`, gradient clipped at 1.0
- Headless training, vectorised `train_long()`

Run it (the default):

```bash
cd RL
python snake_improved_RL.py --no-render --max-games 100 --loss huber
python run_baseline.py --games 100
```

## Results — 250 training games, 100 evaluation episodes each

| Agent | Exp 1 (MSE) eval avg |
| --- | --- |
| Standard | 48.36 |
| Improved | 44.94 |
| Improved Double DQN | 44.00 |

These numbers are Reyes's own 250-game run on his laptop
(Python 3.14, pygame-ce 2.5.8). Single run, standard deviations
around 10. Huber results will be added after the same run is
repeated on the laptop. The clear gain so far is training speed
(about 3× faster) plus the runner.

At 100 games, epsilon is still 30.5 and neither version is
converged, so 100-game runs are not a fair comparison — use 250
games. Note: checkpoints are shared between the two losses, so
running one loss after the other overwrites that experiment's
best model. Evaluation results are timestamped in
`RL/evaluations/`, so they survive.
