"""Run all three Snake trainers, then evaluate each one.

Trains standard, improved and improved Double DQN for the same
number of games, headless, one after another, then runs the
independent greedy evaluation for each experiment.

Usage, from the RL/ directory:

    python run_baseline.py
    python run_baseline.py --games 100 --episodes 100
    python run_baseline.py --games 50 --skip-eval
    python run_baseline.py --games 100 --render
    python run_baseline.py --games 100 --loss mse

Training outputs and evaluation results are saved in the usual
places (RL/checkpoints/, RL/logs/, RL/replays/, RL/evaluations/)
and are not committed. If one trainer fails, the runner prints
the failure and moves on to the next one instead of stopping.

Note: checkpoints are shared between the two losses, so running
with --loss mse after a --loss huber run overwrites that
experiment's best model. Evaluation results are timestamped, so
they survive — check RL/evaluations/ for each run's numbers.
"""

import argparse
import subprocess
import sys
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    ("standard", "snake_standard_RL.py"),
    ("improved", "snake_improved_RL.py"),
    ("improved_double", "snake_improved_double_RL.py"),
]


def run(command, label):
    """Run one command, printing what it is, and return its exit code."""

    print("=" * 60)
    print(label)
    print(" ".join(str(part) for part in command))
    print("=" * 60, flush=True)

    result = subprocess.run(command, cwd=RL_DIR)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Train all three Snake agents and evaluate each one."
    )

    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Training games per agent (default: 100).",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Evaluation episodes per agent (default: 100).",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train only, do not run evaluation afterwards.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the game window while training (default: headless).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each trainer from its latest checkpoint.",
    )

    parser.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="huber",
        help="Loss function: mse (Experiment 1) or huber (Experiment 2, default).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training (default: 42). Same seed + same code = same run.",
    )

    args = parser.parse_args()

    if args.games < 1 or args.episodes < 1:
        parser.error("Games and episodes must be positive.")

    results = {}

    print(f"Loss function: {args.loss} | Seed: {args.seed}", flush=True)

    for experiment, script in EXPERIMENTS:
        command = [
            sys.executable,
            script,
            "--max-games",
            str(args.games),
            "--loss",
            args.loss,
            "--seed",
            str(args.seed),
        ]

        if not args.render:
            command.append("--no-render")

        if args.resume:
            command.append("--resume")

        code = run(command, f"Training: {experiment} ({args.games} games)")
        results[f"train:{experiment}"] = code

        if code != 0:
            print(
                f"WARNING: training {experiment} exited with code {code}. "
                "Continuing with the next experiment.",
                flush=True,
            )

        if args.skip_eval:
            continue

        code = run(
            [
                sys.executable,
                "-m",
                "experiments.evaluate",
                "--experiment",
                experiment,
                "--episodes",
                str(args.episodes),
            ],
            f"Evaluating: {experiment} ({args.episodes} episodes)",
        )
        results[f"eval:{experiment}"] = code

        if code != 0:
            print(
                f"WARNING: evaluation {experiment} exited with code {code}.",
                flush=True,
            )

    print()
    print("=" * 60)
    print(f"BASELINE SUMMARY (loss: {args.loss}, seed: {args.seed})")
    print("=" * 60)

    for label, code in results.items():
        print(f"{label}: {'OK' if code == 0 else f'FAILED (exit {code})'}")

    failed = [label for label, code in results.items() if code != 0]

    if failed:
        print(f"\nFailed steps: {', '.join(failed)}")
        return 1

    print("\nAll steps finished. Check RL/evaluations/ for the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
