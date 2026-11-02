from common.config import GRID


def standard_reward(game, t):
    if t["death_reason"]:
        return -100.0
    reward = -0.2
    reward += 0.5 if t["new_distance"] < t["old_distance"] else -0.2
    if t["ate_food"]:
        reward += 20.0
    return reward


def improved_reward(game, t):
    if t["death_reason"]:
        return -20.0
    reward = -0.2 + 0.5 * (t["old_distance"] - t["new_distance"]) / GRID
    if not t["tail_reachable"] and t["local_space"] < 0.5 * t["global_space"]:
        reward -= 20.0
    reward += 0.05 * t["local_space"]
    if game.position_history.count(game.head_pos) > 2:
        reward -= 2.0
    if t["ate_food"]:
        reward += 20.0
    return reward
