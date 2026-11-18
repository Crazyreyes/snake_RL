# Experiments

Both experiments train the same three agents (standard, improved, improved Double DQN) on the same game, rewards, network (15 → 128 → 3), epsilon (80 → 5, decay 0.5), learning rate (0.001), gamma (0.9), memory (5000) and batch size (128). Only the loss differs.

## Shared code (same for both experiments)

- `RL/run_baseline.py` — trains all three agents, then evaluates each
- `RL/common/renderer.py` and `--no-render` / `SNAKE_RL_RENDER=0` — headless training
- `--max-games` — stop automatically after N games
- `RL/experiments/train_common.py` — shared training loop
- `RL/experiments/evaluate.py` — independent greedy evaluation

## Seeding

Training is seeded with `--seed` (default 42). It seeds Python's `random` (food placement, exploration, replay sampling), numpy and torch (network initial weights), via `RL/common/seed.py`.

Same seed + same code + same machine and torch version = the same training run. Different seeds give different runs — use them to repeat a baseline and average out the luck:

```bash
cd RL
python run_baseline.py --games 250 --loss huber --seed 42
python run_baseline.py --games 250 --loss huber --seed 1
python run_baseline.py --games 250 --loss huber --seed 2
```

The seed is recorded in the training log filename (`snake_improved_huber_seed42_training_...`), in the training CSV's Seed column, in checkpoint metadata, and in the evaluation JSON's `seed` field. Resuming a checkpoint with a different seed (or a different loss) is refused — the checkpoint's seed must match. Seeding does not guarantee bit-for-bit identical results across different machines, torch versions or pygame builds.

## Experiment 1 — MSE loss

- Loss: `nn.MSELoss()`, no gradient clipping
- `train_long()` was originally a per-sample Python loop; the vectorised version in `RL/agents/dqn_agent.py` computes the same targets (verified: max difference 0.00e+00 on the same batch), so it does not change the learning, only the speed.

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

## Experiment 3 — Loop fix

The seeded rerun showed the agents loop: Huber improved had 17 truncated games and Double 15 across 3 seeds, vs 5 and 3 for MSE, and standard averaged approximately 1,000 steps per episode. Two causes were found in the code:

1. `standard_reward` had no revisit penalty at all — only a -0.2 step cost and distance shaping. `improved_reward` had one, but only a flat -2.0 on the 3rd visit to a cell in the last 20 steps.
2. A game had no step limit in training — it only ended on death or a full board, so a looping agent could run to the 10,000-step evaluation cap.

The fix, in `RL/common/rewards.py` and `RL/common/snake_game.py`:

- A shared graduated revisit penalty, applied to both rewards: the 3rd visit to a cell in the last 20 steps costs 2.0, the 4th costs 4.0, and so on. Passing through a cell twice costs nothing.
- A starvation timeout (`STARVATION_LIMIT = 100` in `RL/common/config.py`): if the snake goes 100 steps without eating, the game ends with `death_reason` `"timeout"`, which the reward functions punish like a death. The board is 13x13, so crossing it takes at most approximately 26 steps — 100 is generous. The timeout also ends evaluation episodes early, so truncated counts should fall.

### Loop-fix results — 250 training games, Huber loss

Three-seed averages, comparing the original `main` behaviour against `loop-fix`:

| Agent | Main eval avg | Loop-fix eval avg | Main avg steps | Loop-fix avg steps | Main truncations | Loop-fix truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 54.78 | 56.78 | 997 | 1118 | 5 | 6 |
| Improved | 48.28 | 48.75 | 1251 | 922 | 17 | 8 |
| Improved Double DQN | 48.08 | 46.33 | 1176 | 1039 | 15 | 12 |

The loop-fix reduced average episode length and truncations for the Improved and Improved Double agents. However, it did not consistently improve evaluation scores. Standard scored slightly higher, but its average episode length and truncations increased.

This suggests that reducing repetitive movement and maximising evaluation scores are not necessarily the same objective.

## Results — seeded rerun, 250 training games, 100 evaluation episodes each

Seeded rerun on Reyes's laptop (Python 3.14, pygame-ce 2.5.8, torch 2.14.0 CPU), 11 Nov 2026. Each cell is the mean evaluation average across seeds 42, 1 and 2; the per-seed values are in brackets. Results are the timestamped evaluation JSONs in `RL/evaluations/`, summarised with `RL/summarize_evaluations.py`.

| Agent | Exp 1 (MSE) mean eval avg | Exp 2 (Huber) mean eval avg | Huber − MSE |
| --- | ---: | ---: | ---: |
| Standard | 52.09 (50.27, 53.82, 52.17) | 54.78 (57.94, 53.29, 53.11) | +2.69 |
| Improved | 47.71 (49.66, 47.32, 46.15) | 48.28 (49.20, 45.78, 49.87) | +0.57 |
| Improved Double DQN | 45.95 (45.77, 43.38, 48.70) | 48.08 (50.01, 49.52, 44.72) | +2.13 |

Per-seed values are listed in seed order 42, 1, 2. Standard deviation across seeds: Huber 2.24 / 1.79 / 2.39, MSE 1.45 / 1.46 / 2.18 (standard / improved / Double).

Huber is ahead on all three agents. But the gaps (+2.69, +0.57, +2.13) are about the same size as the seed-to-seed spread, so this is a consistent small edge, not a proven large win — improved is effectively tied (+0.57). Standard is the best agent on both losses. Double DQN does not beat improved on this rerun.

Huber is also loopier: truncated games across the 3 seeds (300 episodes per agent) — Huber improved 17 and Double 15, vs MSE improved 5 and Double 3; standard 5 vs 3.

This seeded rerun supersedes the earlier single runs, which were one run each and could have been luck. The clear gain remains training speed (about 3× faster) plus the runner.

At 100 games, epsilon is still 30.5 and neither version is converged, so 100-game runs are not a fair comparison — use 250 games. Note: checkpoints are shared between the two losses, so running one loss after the other overwrites that experiment's best model. The loss is recorded in the checkpoint itself, in the training log filename (`snake_improved_huber_training_...` / `snake_improved_mse_training_...`), in the training CSV's Loss column, and in the evaluation JSON's `loss` field — so a run can always be identified afterwards. Evaluation results are timestamped in `RL/evaluations/`, so they survive.

## Experiment 4 — Increasing training duration to 1000 games

To investigate whether longer training improves performance, all three agents were trained for 1000 games using the `loop-fix` branch. The same network architecture and other hyperparameters were retained.

### Huber loss — 250 vs 1000 games

Three-seed averages across seeds 42, 1 and 2:

| Agent | 250 games | 1000 games | Difference |
| --- | ---: | ---: | ---: |
| Standard | 56.78 | 46.15 | -10.63 |
| Improved | 48.75 | 46.39 | -2.36 |
| Improved Double DQN | 46.33 | 46.54 | +0.21 |

Per-seed results for 1000 games:

| Agent | Seed 42 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Standard | 45.40 | 42.99 | 50.07 |
| Improved | 49.08 | 47.74 | 42.35 |
| Improved Double DQN | 49.70 | 43.34 | 46.59 |

### MSE loss — 250 vs 1000 games

The 250-game results below are seed 42 only, so this is a single-seed comparison rather than a three-seed average.

| Agent | 250 games, seed 42 | 1000 games, seed 42 | Difference |
| --- | ---: | ---: | ---: |
| Standard | 50.27 | 48.43 | -1.84 |
| Improved | 49.66 | 48.32 | -1.34 |
| Improved Double DQN | 45.77 | 45.48 | -0.29 |

### MSE vs Huber — 1000 games

Three-seed averages across seeds 42, 1 and 2:

| Agent | Huber mean | MSE mean | Huber − MSE |
| --- | ---: | ---: | ---: |
| Standard | 46.15 | 46.38 | -0.23 |
| Improved | 46.39 | 46.09 | +0.30 |
| Improved Double DQN | 46.54 | 47.47 | -0.93 |

Per-seed evaluation averages:

| Loss | Seed | Standard | Improved | Improved Double DQN |
| --- | ---: | ---: | ---: | ---: |
| Huber | 42 | 45.40 | 49.08 | 49.70 |
| Huber | 1 | 42.99 | 47.74 | 43.34 |
| Huber | 2 | 50.07 | 42.35 | 46.59 |
| MSE | 42 | 48.43 | 48.32 | 45.48 |
| MSE | 1 | 45.91 | 44.29 | 47.32 |
| MSE | 2 | 44.81 | 45.65 | 49.61 |

Across the three seeds, neither loss function demonstrated a consistent advantage. Huber performed marginally better for the Improved agent, while MSE performed marginally better for Standard and Improved Double DQN. The differences were small and should not be treated as evidence of a definitive winner.

Increasing training duration from 250 to 1000 games also failed to produce a consistent improvement. The Huber Standard agent's mean evaluation score decreased substantially, while Improved decreased slightly and Improved Double remained approximately unchanged.

One notable result was MSE Improved with seed 2, which had four truncated evaluation episodes and an average of 983.97 steps. The other 17 runs had zero truncated episodes.

## Overall findings

- **Loss function:** Huber showed a small advantage at 250 games, but neither Huber nor MSE consistently outperformed the other at 1000 games.
- **Training duration:** Increasing training from 250 to 1000 games did not consistently improve evaluation performance.
- **Double DQN:** Improved Double DQN did not consistently outperform Improved DQN.
- **Loop prevention:** The loop-fix reduced truncations for the Improved agents at 250 games and eliminated truncations in the 1000-game Huber runs. However, one MSE Improved seed 2 run still had four truncated episodes.
- **Evaluation performance:** The agents' scores remained relatively similar across configurations, suggesting that simply changing the loss function or increasing training duration is insufficient to produce a substantial improvement.

These results suggest that future experiments should investigate how efficiently rewards are propagated and how the agent learns from its experience, rather than relying solely on longer training or different loss functions.

## Future experiments

| Priority | Experiment | Purpose |
| ---: | --- | --- |
| 1 | N-step returns (n=3) | Improve reward propagation across multiple transitions |
| 2 | Prioritised Experience Replay | Sample more informative experiences more frequently |
| 3 | Dueling DQN | Improve state-value and action-advantage estimation |
| 4 | Richer state representation | Provide additional information about the environment |
| 5 | Gamma tuning | Investigate the effect of longer-term reward weighting |

### N-step returns (planned)

Investigate whether accumulating rewards across multiple consecutive transitions improves learning efficiency.

The initial implementation will use n=3 while retaining the existing network architecture and other hyperparameters.

The experiment will compare:

- One-step returns (baseline)
- Three-step returns (experimental)

The objective is to determine whether propagating food rewards over multiple steps improves evaluation performance.