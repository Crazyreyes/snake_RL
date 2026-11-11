"""Seeding helper for the random sources used in training."""

import random

import numpy as np
import torch


def seed_all(seed):
    """Seed random, numpy and torch (including CUDA, if present)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
