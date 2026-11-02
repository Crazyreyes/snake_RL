import random
from collections import deque
import numpy as np
from common.config import GRID, PLAY_RECT, TOTAL_CELLS, LEFT, RIGHT


class SnakeGame:
    def reset(self):
        self.direction = (GRID, 0)
        centre = round((680 // 2) / GRID) * GRID
        self.head_pos = (centre, centre)
        self.body_pos = []
        self.score = 0
        self.spawn_food()
        self.frame = 0
        self.position_history = deque(maxlen=20)
        return self.get_state()

    def __init__(self):
        self.reset()

    def spawn_food(self):
        left, top, width, height = PLAY_RECT
        cells = [
            (x, y)
            for x in range(left, left + width, GRID)
            for y in range(top, top + height, GRID)
            if (x, y) != self.head_pos and (x, y) not in self.body_pos
        ]
        self.food = random.choice(cells) if cells else None
        return self.food is not None

    def collision_type(self, x, y, tail_will_move=False):
        left, top, width, height = PLAY_RECT
        if not (left <= x < left + width and top <= y < top + height):
            return "wall"
        body = self.body_pos[:-1] if tail_will_move and self.body_pos else self.body_pos
        return "body" if (x, y) in body else None

    def is_danger(self, pos):
        return self.collision_type(*pos, tail_will_move=(pos != self.food)) is not None

    def flood_fill_space(self, start):
        q = deque([start])
        seen = set()
        growing = start == self.food
        blocked = set(
            self.body_pos[:-1] if self.body_pos and not growing else self.body_pos
        )
        left, top, width, height = PLAY_RECT
        right, bottom = left + width, top + height
        while q:
            p = q.popleft()
            x, y = p
            if (
                p in seen
                or not (left <= x < right and top <= y < bottom)
                or p in blocked
            ):
                continue
            if p == self.head_pos and p != start:
                continue
            seen.add(p)
            q.extend([(x + GRID, y), (x - GRID, y), (x, y + GRID), (x, y - GRID)])
        return len(seen)

    def get_global_free_space(self, tail_will_move=False):
        occupied = (
            1 + len(self.body_pos) - (1 if tail_will_move and self.body_pos else 0)
        )
        return max(0, TOTAL_CELLS - occupied) / TOTAL_CELLS

    def can_reach_tail(self, head_pos, tail_will_move=True):
        if not self.body_pos:
            return True
        tail = self.body_pos[-1]
        blocked = set(self.body_pos[:-1] if tail_will_move else self.body_pos)
        q = deque([head_pos])
        seen = set()
        left, top, width, height = PLAY_RECT
        right, bottom = left + width, top + height
        while q:
            p = q.popleft()
            x, y = p
            if (
                p in seen
                or not (left <= x < right and top <= y < bottom)
                or p in blocked
            ):
                continue
            seen.add(p)
            if tail_will_move and p == tail:
                return True
            if not tail_will_move and any(
                (x + dx, y + dy) == tail
                for dx, dy in [(GRID, 0), (-GRID, 0), (0, GRID), (0, -GRID)]
            ):
                return True
            q.extend([(x + GRID, y), (x - GRID, y), (x, y + GRID), (x, y - GRID)])
        return False

    def get_state(self):
        hx, hy = self.head_pos
        dx, dy = self.direction
        l = (-dy, dx)
        r = (dy, -dx)
        positions = [(hx + dx, hy + dy), (hx + l[0], hy + l[1]), (hx + r[0], hy + r[1])]
        danger = [self.is_danger(p) for p in positions]
        spaces = [self.flood_fill_space(p) / TOTAL_CELLS for p in positions]
        food = self.food if self.food is not None else self.head_pos
        return np.array(
            [
                *danger,
                *spaces,
                self.get_global_free_space(),
                dx < 0,
                dx > 0,
                dy < 0,
                dy > 0,
                food[0] < hx,
                food[0] > hx,
                food[1] < hy,
                food[1] > hy,
            ],
            dtype=np.float32,
        )

    def move(self, action):
        dirs = [(GRID, 0), (0, GRID), (-GRID, 0), (0, -GRID)]
        i = dirs.index(self.direction)
        if action == LEFT:
            i = (i - 1) % 4
        elif action == RIGHT:
            i = (i + 1) % 4
        self.direction = dirs[i]

    def step(self, action, reward_function):
        self.frame += 1
        old = self.head_pos
        old_food = self.food
        old_dist = (
            abs(old[0] - old_food[0]) + abs(old[1] - old_food[1]) if old_food else 0
        )
        self.move(action)
        dx, dy = self.direction
        new = (old[0] + dx, old[1] + dy)
        grow = new == self.food
        death = self.collision_type(*new, tail_will_move=not grow)
        if death:
            t = {
                "old_head": old,
                "new_head": new,
                "old_distance": old_dist,
                "new_distance": old_dist,
                "ate_food": False,
                "death_reason": death,
                "local_space": 0.0,
                "global_space": self.get_global_free_space(),
                "tail_reachable": False,
            }
            return reward_function(self, t), True, self.score, t
        self.body_pos.insert(0, old)
        self.head_pos = new
        if grow:
            self.score += 1
            self.spawn_food()
        else:
            self.body_pos.pop()
        new_dist = (
            abs(new[0] - self.food[0]) + abs(new[1] - self.food[1]) if self.food else 0
        )
        local = self.flood_fill_space(new) / TOTAL_CELLS
        global_space = self.get_global_free_space()
        reachable = self.can_reach_tail(new, tail_will_move=not grow)
        self.position_history.append(new)
        t = {
            "old_head": old,
            "new_head": new,
            "old_distance": old_dist,
            "new_distance": new_dist,
            "ate_food": grow,
            "death_reason": None,
            "local_space": local,
            "global_space": global_space,
            "tail_reachable": reachable,
        }
        return reward_function(self, t), self.food is None, self.score, t
