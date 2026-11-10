import argparse

from common.rewards import improved_reward
from experiments.train_common import run_training

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the Improved Double DQN Snake agent."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest training checkpoint.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save a checkpoint every N completed games.",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Train headless, without opening a game window.",
    )

    args = parser.parse_args()

    run_training(
        improved_reward,
        "improved_double",
        double_dqn=True,
        epsilon_start=80,
        epsilon_decay=0.5,
        epsilon_min=5,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        render=not args.no_render,
    )
