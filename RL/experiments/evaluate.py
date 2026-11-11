"""Independent evaluation for trained Snake DQN models."""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

import torch

from common.checkpoint import (
    get_checkpoint_paths,
    load_model_for_inference,
)
from common.config import PROJECT_DIR
from common.snake_game import SnakeGame

EVALUATION_DIR = PROJECT_DIR / "evaluations"


def zero_reward(game, transition):
    """Evaluation does not use rewards for learning."""

    return 0.0


def evaluate_policy(
    model,
    episodes=100,
    seed_start=10000,
    max_steps=10000,
):
    """Evaluate a trained model without exploration or training."""

    if episodes < 1:
        raise ValueError("Episodes must be at least 1.")

    if max_steps < 1:
        raise ValueError("Maximum steps must be at least 1.")

    model.eval()

    results = []

    for episode in range(episodes):

        seed = seed_start + episode
        random.seed(seed)

        game = SnakeGame()

        steps = 0
        terminated = False

        while steps < max_steps:

            state = game.get_state()

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
            )

            # Greedy action selection.
            # No random exploration and no gradient updates.
            with torch.inference_mode():

                q_values = model(state_tensor)

                action = int(torch.argmax(q_values).item())

            _, done, score, _ = game.step(
                action,
                zero_reward,
            )

            steps += 1

            if done:
                terminated = True
                break

        result = {
            "episode": episode + 1,
            "seed": seed,
            "score": int(game.score),
            "steps": steps,
            "terminated": terminated,
            "truncated": not terminated,
        }

        results.append(result)

        if (episode + 1) % 10 == 0:
            print(
                f"Evaluated {episode + 1}/{episodes} games | "
                f"Latest score: {game.score}"
            )

    return results


def calculate_metrics(results):
    """Calculate evaluation statistics."""

    scores = [r["score"] for r in results]
    steps = [r["steps"] for r in results]

    return {
        "games": len(results),
        "average_score": round(mean(scores), 2),
        "median_score": median(scores),
        "score_std_dev": round(pstdev(scores), 2),
        "best_score": max(scores),
        "worst_score": min(scores),
        "average_steps": round(mean(steps), 2),
        "games_score_10_or_more": sum(score >= 10 for score in scores),
        "games_score_20_or_more": sum(score >= 20 for score in scores),
        "games_score_30_or_more": sum(score >= 30 for score in scores),
        "truncated_games": sum(result["truncated"] for result in results),
    }


def save_results(
    experiment_name,
    checkpoint_path,
    results,
    metrics,
    loss="unknown",
    seed="unknown",
):
    """Save episode results and evaluation summary."""

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix = f"snake_{experiment_name}_evaluation_{timestamp}"

    csv_path = EVALUATION_DIR / f"{prefix}.csv"
    json_path = EVALUATION_DIR / f"{prefix}.json"

    # Save individual game results.
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "seed",
                "score",
                "steps",
                "terminated",
                "truncated",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    # Save complete evaluation summary.
    summary = {
        "experiment": experiment_name,
        "loss": loss,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "evaluation_timestamp": datetime.now().isoformat(),
        "seed_start": results[0]["seed"],
        "seed_end": results[-1]["seed"],
        "metrics": metrics,
        "episodes": results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return csv_path, json_path


def main():
    """Parse arguments, evaluate a trained model and save the results."""

    parser = argparse.ArgumentParser(
        description=("Independently evaluate a trained Snake DQN model.")
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="improved",
        help="Experiment name used when saving the checkpoint.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a specific inference checkpoint.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation games.",
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=10000,
        help="Starting random seed for evaluation.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum steps allowed per evaluation game.",
    )

    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1:
        parser.error("Episodes and maximum steps must be positive.")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        _, checkpoint_path = get_checkpoint_paths(args.experiment)

    print("=" * 55)
    print("       SNAKE RL INDEPENDENT EVALUATION")
    print("=" * 55)

    print("Experiment:", args.experiment)
    print("Checkpoint:", checkpoint_path)
    print("Evaluation games:", args.episodes)
    print("Starting seed:", args.seed_start)

    # Load inference-only model.
    model, metadata = load_model_for_inference(
        checkpoint_path,
        map_location="cpu",
    )

    print(
        "Model training experiment:",
        metadata["experiment_name"],
    )

    print(
        "Best training score:",
        metadata["score"],
    )

    print(
        "Loss:",
        metadata.get("loss", "unknown"),
    )

    print(
        "Seed:",
        metadata.get("seed", "unknown"),
    )

    print("\nStarting evaluation...\n")

    results = evaluate_policy(
        model=model,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
    )

    metrics = calculate_metrics(results)

    csv_path, json_path = save_results(
        args.experiment,
        checkpoint_path,
        results,
        metrics,
        loss=metadata.get("loss", "unknown"),
        seed=metadata.get("seed", "unknown"),
    )

    print("\n" + "=" * 55)
    print("             EVALUATION RESULTS")
    print("=" * 55)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nResults saved:")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
