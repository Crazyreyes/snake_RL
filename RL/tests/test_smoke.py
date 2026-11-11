"""
===============================================================================
test_smoke.py
===============================================================================

Description:
    Smoke tests for the Snake RL trainers, as pytest.

    The suite covers:
    1. The game state has 15 float32 values.
    2. The same seed gives the same random numbers, and a different seed differs.
    3. The revisit penalty is 0 for one or two visits, -2.0 for three visits and -4.0 for four.
    4. Going 100 steps without food ends the game as a timeout, for both the standard and improved rewards.
    5. Eating food resets the starvation counter.
    6. Training the improved agent for 3 games and evaluating it for 1 episode works end to end.

    Testing approach
    ----------------
    - The fast tests check the game, the rewards and the seeding directly.
    - The end-to-end test trains the improved agent for 3 games headlessly in a temporary copy of RL/ and evaluates it for 1 episode, exactly like the manual smoke test — it just does it in tmp_path, so it never touches your real logs, checkpoints or evaluations.
    - The end-to-end test is marked slow, so it can be skipped with -m "not slow".

    Run the test suite from the RL/ directory using:
        python -m pytest tests/ -v
    or for the fast tests only:
        python -m pytest tests/ -v -m "not slow"
"""

import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from common.rewards import improved_reward, revisit_penalty, standard_reward
from common.seed import seed_all
from common.snake_game import SnakeGame

RL_DIR = Path(__file__).resolve().parent.parent


def test_state_shape():
    """Check that the game state has 15 float32 values."""

    game = SnakeGame()
    state = game.get_state()
    assert state.shape == (15,)
    assert state.dtype.name == "float32"


def test_seed_repeatable():
    """Check that the same seed gives the same random numbers."""

    seed_all(42)
    first = [random.random() for _ in range(5)]
    seed_all(42)
    second = [random.random() for _ in range(5)]
    assert first == second
    seed_all(7)
    assert [random.random() for _ in range(5)] != first


def test_revisit_penalty():
    """Check the revisit penalty for one, two, three and four visits."""

    game = SnakeGame()
    game.head_pos = (100, 100)
    game.position_history.extend([(100, 100)] * 1)
    assert revisit_penalty(game) == 0.0  # 1 visit: normal
    game.position_history.append((100, 100))
    assert revisit_penalty(game) == 0.0  # 2 visits: still free
    game.position_history.append((100, 100))
    assert revisit_penalty(game) == -2.0  # 3rd visit
    game.position_history.append((100, 100))
    assert revisit_penalty(game) == -4.0  # 4th visit


def test_starvation_timeout():
    """Check that the standard agent times out after 100 steps without food."""

    game = SnakeGame()
    game.steps_since_food = 100
    reward, done, _score, info = game.step(0, standard_reward)
    assert done is True
    assert info["death_reason"] == "timeout"
    assert reward == -100.0  # standard reward punishes death


def test_starvation_timeout_improved_reward():
    """Check that the improved agent times out after 100 steps without food."""

    game = SnakeGame()
    game.steps_since_food = 100
    reward, done, _score, info = game.step(0, improved_reward)
    assert done is True
    assert info["death_reason"] == "timeout"
    assert reward == -20.0  # improved reward punishes death


def test_eating_resets_starvation_counter():
    """Check that eating food resets the starvation counter."""
    
    game = SnakeGame()
    game.food = (game.head_pos[0] + 40, game.head_pos[1])
    game.steps_since_food = 90
    _reward, done, score, info = game.step(0, standard_reward)
    assert info["ate_food"] is True
    assert done is False
    assert score == 1
    assert game.steps_since_food == 0


@pytest.mark.slow
def test_train_and_evaluate_improved(tmp_path):
    """End-to-end smoke test: 3 games, then a 1-episode evaluation."""
    work = tmp_path / "RL"
    shutil.copytree(
        RL_DIR,
        work,
        ignore=shutil.ignore_patterns(
            "__pycache__", "logs", "checkpoints", "evaluations", "replays", "tests"
        ),
    )
    env = dict(os.environ)
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SNAKE_RL_RENDER"] = "0"

    train = subprocess.run(
        [
            sys.executable,
            "snake_improved_RL.py",
            "--no-render",
            "--max-games",
            "3",
            "--loss",
            "huber",
            "--seed",
            "42",
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    assert "Reached max games (3)" in train.stdout

    logs = sorted((work / "logs").glob("*_huber_seed42_training_*.csv"))
    assert logs, "no seed-42 training log was written"
    with logs[0].open(encoding="utf-8") as file:
        scores = [row["Score"] for row in csv.DictReader(file)]
    assert scores == ["0", "1", "0"], scores  # seed 42 repeat

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.evaluate",
            "--experiment",
            "improved",
            "--episodes",
            "1",
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert evaluate.returncode == 0, evaluate.stdout + evaluate.stderr
    assert "Loss: huber" in evaluate.stdout
    assert "Seed: 42" in evaluate.stdout

    jsons = sorted((work / "evaluations").glob("*.json"))
    assert jsons, "no evaluation JSON was written"
    summary = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert summary["loss"] == "huber"
    assert summary["seed"] == 42
