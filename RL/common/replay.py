import json
from datetime import datetime

from common.artifact_naming import experiment_file_prefix
from common.config import REPLAY_DIR


class ReplayRecorder:
    def __init__(self):
        self.frames = []

    def reset(self, game):
        self.frames = []
        self.record(game)

    def record(
        self,
        game,
        action=None,
        reward=0,
        done=False,
        death_reason=None,
        attempted_head=None,
    ):
        self.frames.append(
            {
                "head": list(game.head_pos),
                "body": [list(p) for p in game.body_pos],
                "food": list(game.food) if game.food is not None else None,
                "direction": list(game.direction),
                "score": game.score,
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "death_reason": death_reason,
                "attempted_head": (
                    list(attempted_head) if attempted_head is not None else None
                ),
            }
        )

    def save(self, game_number, score, experiment_name):
        """Save a replay using the same experiment prefix as its training run."""
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)

        prefix = experiment_file_prefix(experiment_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            f"{prefix}_game_{game_number:04d}_score_{score}_{stamp}.json"
        )
        path = REPLAY_DIR / filename

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "game_number": game_number,
                    "score": score,
                    "timestamp": datetime.now().isoformat(timespec="microseconds"),
                    "frames": self.frames,
                },
                file,
                indent=2,
            )

        return path
