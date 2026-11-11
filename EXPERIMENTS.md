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
 
## Seeding

Training is seeded with `--seed` (default 42). It seeds Python's
`random` (food placement, exploration, replay sampling), numpy
and torch (network initial weights), via `RL/common/seed.py`.

Same seed + same code + same machine and torch version = the same
training run. Different seeds give different runs — use them to
repeat a baseline and average out the luck:

```bash
cd RL
python run_baseline.py --games 250 --loss huber --seed 42
python run_baseline.py --games 250 --loss huber --seed 1
python run_baseline.py --games 250 --loss huber --seed 2
```

The seed is recorded in the training log filename
(`snake_improved_huber_seed42_training_...`), in the training
CSV's Seed column, in checkpoint metadata, and in the evaluation
JSON's `seed` field. Resuming a checkpoint with a different seed
(or a different loss) is refused — the checkpoint's seed must
match. Seeding does not guarantee bit-for-bit identical results
across different machines, torch versions or pygame builds.

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

## Loop fix (Experiment 3)

The seeded rerun showed the agents loop: Huber improved had 17
truncated games and Double 15 across 3 seeds, vs 5 and 3 for MSE,
and standard averaged ~1,000 steps per episode. Two causes were
found in the code:

1. `standard_reward` had no revisit penalty at all — only a
   -0.2 step cost and distance shaping. `improved_reward` had
   one, but only a flat -2.0 on the 3rd visit to a cell in the
   last 20 steps.
2. A game had no step limit in training — it only ended on death
   or a full board, so a looping agent could run to the 10,000-step
   evaluation cap.

The fix, in `RL/common/rewards.py` and `RL/common/snake_game.py`:

- A shared graduated revisit penalty, applied to both rewards:
  the 3rd visit to a cell in the last 20 steps costs 2.0, the
  4th costs 4.0, and so on. Passing through a cell twice costs
  nothing.
- A starvation timeout (`STARVATION_LIMIT = 100` in
  `RL/common/config.py`): if the snake goes 100 steps without
  eating, the game ends with `death_reason` `"timeout"`, which
  the reward functions punish like a death. The board is 13x13,
  so crossing it takes at most ~26 steps — 100 is generous.
  The timeout also ends evaluation episodes early, so truncated
  counts should fall.

Measure it the same seeded way as the Huber vs MSE rerun:
`run_baseline.py --games 250 --loss huber --seed 42/1/2` (and
MSE for comparison). Compare eval averages, average steps and
truncated games against the seeded rerun table below.

## Results — seeded rerun, 250 training games, 100 evaluation episodes each

Seeded rerun on Reyes's laptop (Python 3.14, pygame-ce 2.5.8,
torch 2.14.0 CPU), 11 Nov 2026. Each cell is the mean evaluation
average across seeds 42, 1 and 2; the per-seed values are in
brackets. Results are the timestamped evaluation JSONs in
`RL/evaluations/`, summarised with `RL/summarize_evaluations.py`.

| Agent | Exp 1 (MSE) mean eval avg | Exp 2 (Huber) mean eval avg | Huber − MSE |
| --- | --- | --- | --- |
| Standard | 52.09 (50.27, 53.82, 52.17) | 54.78 (57.94, 53.29, 53.11) | +2.69 |
| Improved | 47.71 (49.66, 47.32, 46.15) | 48.28 (49.20, 45.78, 49.87) | +0.57 |
| Improved Double DQN | 45.95 (45.77, 43.38, 48.70) | 48.08 (50.01, 49.52, 44.72) | +2.13 |

Per-seed values are listed in seed order 42, 1, 2. Standard
deviation across seeds: Huber 2.24 / 1.79 / 2.39, MSE 1.45 /
1.46 / 2.18 (standard / improved / Double).

Huber is ahead on all three agents. But the gaps
(+2.69, +0.57, +2.13) are about the same size as
the seed-to-seed spread, so this is a consistent small edge,
not a proven large win — improved is effectively tied (+0.57).
Standard is the best agent on both losses. Double DQN does not
beat improved on this rerun.

Huber is also loopier: truncated games across the 3 seeds
(300 episodes per agent) — Huber improved 17 and Double 15,
vs MSE improved 5 and Double 3; standard 5 vs 3.

This seeded rerun supersedes the earlier single runs, which
were one run each and could have been luck. The clear gain
remains training speed (about 3× faster) plus the runner.

At 100 games, epsilon is still 30.5 and neither version is
converged, so 100-game runs are not a fair comparison — use 250
games. Note: checkpoints are shared between the two losses, so
running one loss after the other overwrites that experiment's
best model. The loss is recorded in the checkpoint itself, in the
training log filename (`snake_improved_huber_training_...` /
`snake_improved_mse_training_...`), in the training CSV's Loss
column, and in the evaluation JSON's `loss` field — so a run can
always be identified afterwards. Evaluation results are
timestamped in `RL/evaluations/`, so they survive.
