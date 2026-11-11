"""
===============================================================================
COMMON.CONFIG
===============================================================================

Description:
    Shared paths, board settings and training constants for Snake RL.

    Board: 13 x 13 cells on a 520 x 520 play area. Training:
    memory 5000, batch 128, learning rate 0.001, gamma 0.9,
    epsilon 80 down to 5. Looping: starvation limit 100 steps
    without food, which counts as a death.

===============================================================================
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_DIR.parent / "improved_assets"
LOG_DIR = PROJECT_DIR / "logs"
REPLAY_DIR = PROJECT_DIR / "replays"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

GRID, WINDOW_SIZE, MARGIN, FPS = 40, 680, 80, 60
RENDER = os.environ.get("SNAKE_RL_RENDER", "1") not in (
    "0",
    "false",
    "False",
    "no",
)

PLAY_RECT = (
    MARGIN,
    MARGIN,
    WINDOW_SIZE - 2 * MARGIN,
    WINDOW_SIZE - 2 * MARGIN,
)

TOTAL_CELLS = ((WINDOW_SIZE - 2 * MARGIN) // GRID) ** 2

STRAIGHT, LEFT, RIGHT = 0, 1, 2

MAX_MEMORY, BATCH_SIZE = 5000, 128
STARVATION_LIMIT = 100
LEARNING_RATE, GAMMA = 0.001, 0.9

LIGHTGREEN = (170, 215, 81)
GREEN = (0, 255, 0)
DARKGREEN = (167, 209, 61)
DARKERGREEN = (10, 50, 10)
RED = (255, 0, 0)
CREAM = (255, 253, 208)
BLACK = (0, 0, 0)
