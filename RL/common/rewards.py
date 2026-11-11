"""Reward functions for the standard and improved Snake agents."""

from common.config import GRID


def revisit_penalty(game):
    """Return the penalty for revisiting the current head position."""

    visits = game.position_history.count(game.head_pos)
    return -2.0 * (visits - 2) if visits > 2 else 0.0


def standard_reward(game, t):
    """Return the standard reward for one game step."""

    if t["death_reason"]:
        return -100.0
    reward = -0.2
    reward += 0.5 if t["new_distance"] < t["old_distance"] else -0.2
    reward += revisit_penalty(game)
    if t["ate_food"]:
        reward += 20.0
    return reward


def improved_reward(game, t):
    """Return the improved reward for one game step."""

    if t["death_reason"]:
        return -20.0
    reward = -0.2 + 0.5 * (t["old_distance"] - t["new_distance"]) / GRID
    if not t["tail_reachable"] and t["local_space"] < 0.5 * t["global_space"]:
        reward -= 20.0
    reward += 0.05 * t["local_space"]
    reward += revisit_penalty(game)
    if t["ate_food"]:
        reward += 20.0
    return reward
