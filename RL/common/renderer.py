"""Pygame renderer that draws the Snake game during training."""

import pygame
from common.config import (
    GRID,
    WINDOW_SIZE,
    PLAY_RECT,
    FPS,
    RENDER,
    ASSET_DIR,
    LIGHTGREEN,
    GREEN,
    DARKGREEN,
    DARKERGREEN,
    MARGIN,
    CREAM,
)


class SnakeRenderer:
    """Pygame window that draws the Snake game."""

    def __init__(self, title="Snake RL", enabled=None):
        """Set up the display window and load the game assets if rendering is enabled."""

        self.enabled = RENDER if enabled is None else enabled
        if not self.enabled:
            return
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("gabriola", 48)
        self.load_assets()

    def image(self, name):
        """Load one sprite image and scale it to a single grid cell."""

        img = pygame.image.load(str(ASSET_DIR / name)).convert_alpha()
        return pygame.transform.scale(img, (GRID, GRID))

    def load_assets(self):
        """Load the apple, head, body and tail sprites."""

        self.apple = self.image("apple.png")
        self.head = {
            d: self.image(f"head_{d}.png") for d in ("up", "down", "left", "right")
        }
        self.body = {
            k: self.image(v)
            for k, v in {
                "vertical": "body_vertical.png",
                "horizontal": "body_horizontal.png",
                "tl": "body_topleft.png",
                "tr": "body_topright.png",
                "bl": "body_bottomleft.png",
                "br": "body_bottomright.png",
            }.items()
        }
        self.tail = {
            d: self.image(f"tail_{d}.png") for d in ("up", "down", "left", "right")
        }

    def render(self, g):
        """Draw the current game state to the screen."""

        if not self.enabled:
            return
        self.screen.fill(CREAM)
        pygame.draw.rect(self.screen, DARKERGREEN, PLAY_RECT, 4)
        l, t, w, h = PLAY_RECT
        for y in range(t, t + h, GRID):
            for x in range(l, l + w, GRID):
                c = LIGHTGREEN if ((x // GRID + y // GRID) % 2) == 0 else DARKGREEN
                pygame.draw.rect(self.screen, c, (x, y, GRID, GRID))
        # Draw title within the top margin.
        title = self.font.render("~~Snake~~", True, GREEN)

        title_rect = title.get_rect(center=(WINDOW_SIZE // 2, MARGIN // 2))

        self.screen.blit(title, title_rect)
        self.screen.blit(self.font.render(f"Score: {g.score}", True, GREEN), (10, 10))
        if g.food is not None:
            self.screen.blit(self.apple, g.food)
        dx, dy = g.direction
        d = "right" if dx > 0 else "left" if dx < 0 else "down" if dy > 0 else "up"
        self.screen.blit(self.head[d], g.head_pos)
        seg = [g.head_pos] + g.body_pos
        for i in range(1, len(seg) - 1):
            p, c, n = seg[i - 1], seg[i], seg[i + 1]
            px, py = p
            cx, cy = c
            nx, ny = n
            if px == nx:
                key = "vertical"
            elif py == ny:
                key = "horizontal"
            elif (px < cx and ny < cy) or (nx < cx and py < cy):
                key = "tl"
            elif (px > cx and ny < cy) or (nx > cx and py < cy):
                key = "tr"
            elif (px < cx and ny > cy) or (nx < cx and py > cy):
                key = "bl"
            else:
                key = "br"
            self.screen.blit(self.body[key], c)
        if g.body_pos:
            tail = g.body_pos[-1]
            before = g.body_pos[-2] if len(g.body_pos) > 1 else g.head_pos
            bx, by = before
            tx, ty = tail
            d = (
                "left"
                if bx > tx
                else "right" if bx < tx else "up" if by > ty else "down"
            )
            self.screen.blit(self.tail[d], tail)
        pygame.display.flip()
        self.clock.tick(FPS)
