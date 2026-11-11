import csv
import logging
from datetime import datetime

from common.artifact_naming import experiment_file_prefix
from common.config import LOG_DIR

CSV_HEADER = [
    "Timestamp",
    "Loss",
    "Seed",
    "Game",
    "Score",
    "Epsilon",
    "Steps",
    "Total Reward",
    "Replay Saved",
]


def setup_logger(name, loss="huber", seed=42):
    """Create a unique CSV and log file for one training session."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    prefix = experiment_file_prefix(name)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{prefix}_{loss}_seed{seed}_training_{run_stamp}"

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Close any handlers left behind if this run name is reused.
    for existing_handler in logger.handlers[:]:
        logger.removeHandler(existing_handler)
        existing_handler.close()

    log_path = LOG_DIR / f"{run_name}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    csv_path = LOG_DIR / f"{run_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(CSV_HEADER)

    logger.info("Experiment: %s | Loss: %s | Seed: %s", name, loss, seed)

    return logger, csv_path


def log_game(
    logger,
    path,
    game,
    score,
    epsilon,
    steps,
    total_reward,
    replay_saved,
    loss="huber",
    seed=42,
):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                stamp,
                loss,
                seed,
                game,
                score,
                round(epsilon, 2),
                steps,
                round(total_reward, 2),
                replay_saved,
            ]
        )

    logger.info(
        "Game %s | Score %s | Epsilon %.2f | Steps %s | Reward %.2f | Replay %s",
        game,
        score,
        epsilon,
        steps,
        total_reward,
        replay_saved,
    )
