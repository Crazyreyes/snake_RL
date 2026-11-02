import json
from datetime import datetime
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

    def save(self, game_number, score):
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = REPLAY_DIR / f"snake_game_{game_number:04d}_score_{score}_{stamp}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "game_number": game_number,
                    "score": score,
                    "timestamp": datetime.now().isoformat(timespec="microseconds"),
                    "frames": self.frames,
                },
                f,
                indent=2,
            )
        return path
