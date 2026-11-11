"""
===============================================================================
TESTS.CONFTEST
===============================================================================

Description:
    Pytest setup for the RL smoke tests.
 
    Run from the RL/ directory::
 
        python -m pytest tests/ -v

    Adds RL/ to sys.path so `common.*` and `experiments.*` import
    the same way the trainers do, and forces headless pygame so no
    game window opens.

===============================================================================
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SNAKE_RL_RENDER", "0")

RL_DIR = Path(__file__).resolve().parent.parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))
