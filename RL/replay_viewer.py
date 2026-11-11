"""
===============================================================================
REPLAY_VIEWER
===============================================================================

Description:
    Pygame viewer for playing back saved Snake replay files.

    Loads a replay file and draws the recorded frames, so a game
    can be watched back after training has finished.

===============================================================================
"""

import pygame
import json
import os
import sys

# ==========================
# CONFIG
# ==========================

GRID = 40
WINDOW_SIZE = 680
MARGIN = 80

FPS = 60

REPLAY_DIR = "replays"
ASSET_DIR = "../improved_assets"

PLAY_RECT = pygame.Rect(
    MARGIN,
    MARGIN,
    WINDOW_SIZE - MARGIN * 2,
    WINDOW_SIZE - MARGIN * 2,
)

# Playback speeds (frames per second)
PLAYBACK_SPEEDS = [1, 2, 4, 8, 15, 30]

# Colours
LIGHTGREEN = (170, 215, 81)
DARKGREEN = (167, 209, 61)
DARKERGREEN = (10, 50, 10)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DARKRED = (150, 0, 0)
CREAM = (255, 253, 208)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)


# ==========================
# REPLAY VIEWER
# ==========================


class ReplayViewer:
    """Pygame application that plays back saved Snake replays."""

    def __init__(self):
        """Set up the window, load the assets and open the newest replay."""

        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

        pygame.display.set_caption("Snake RL Replay Viewer")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("gabriola", 32)
        self.small_font = pygame.font.SysFont("arial", 16)
        self.title_font = pygame.font.SysFont("gabriola", 42)

        # Replay controls
        self.paused = False
        self.playback_speed_index = 2
        self.current_frame = 0
        self.last_frame_time = pygame.time.get_ticks()

        # Load assets
        self.load_assets()

        # Find available replays
        self.replay_files = self.find_replays()

        self.current_replay_index = 0
        self.replay_data = None
        self.frames = []

        if self.replay_files:
            self.load_replay(0)

    # ==========================
    # ASSETS
    # ==========================

    def load_image(self, filename):
        """Load one sprite image, returning None if the file cannot be loaded."""

        path = os.path.join(ASSET_DIR, filename)

        try:
            image = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(image, (GRID, GRID))

        except pygame.error:
            print(f"Could not load asset: {path}")
            return None

    def load_assets(self):
        """Load the apple, head, body and tail sprites."""

        self.apple = self.load_image("apple.png")

        self.head = {
            "up": self.load_image("head_up.png"),
            "down": self.load_image("head_down.png"),
            "left": self.load_image("head_left.png"),
            "right": self.load_image("head_right.png"),
        }

        self.body = {
            "vertical": self.load_image("body_vertical.png"),
            "horizontal": self.load_image("body_horizontal.png"),
            "tl": self.load_image("body_topleft.png"),
            "tr": self.load_image("body_topright.png"),
            "bl": self.load_image("body_bottomleft.png"),
            "br": self.load_image("body_bottomright.png"),
        }

        self.tail = {
            "up": self.load_image("tail_up.png"),
            "down": self.load_image("tail_down.png"),
            "left": self.load_image("tail_left.png"),
            "right": self.load_image("tail_right.png"),
        }

    # ==========================
    # REPLAY FILES
    # ==========================

    def find_replays(self):
        """Return the saved replay files, newest first."""

        if not os.path.exists(REPLAY_DIR):
            os.makedirs(REPLAY_DIR)

        files = [
            filename
            for filename in os.listdir(REPLAY_DIR)
            if filename.endswith(".json")
        ]

        files.sort(
            key=lambda filename: os.path.getmtime(os.path.join(REPLAY_DIR, filename)),
            reverse=True,
        )

        return files

    def load_replay(self, index):
        """Load the replay at the given index."""

        if not self.replay_files:
            return

        self.current_replay_index = index % len(self.replay_files)

        filename = self.replay_files[self.current_replay_index]
        filepath = os.path.join(REPLAY_DIR, filename)

        try:
            with open(filepath, "r") as file:
                self.replay_data = json.load(file)

            self.frames = self.replay_data.get("frames", [])

            if not self.frames:
                print(f"Replay has no frames: {filename}")
                self.replay_data = None
                return

            self.current_frame = 0
            self.paused = False
            self.last_frame_time = pygame.time.get_ticks()

            print(f"Loaded replay: {filename}")

        except (OSError, json.JSONDecodeError) as error:
            print(f"Could not load replay: {error}")
            self.replay_data = None
            self.frames = []

    # ==========================
    # PLAYBACK CONTROLS
    # ==========================

    def handle_input(self):
        """Handle keyboard input and return whether playback should continue."""

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    return False

                # Pause / Resume
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                # Previous replay
                elif event.key == pygame.K_LEFT:
                    self.load_replay(self.current_replay_index - 1)

                # Next replay
                elif event.key == pygame.K_RIGHT:
                    self.load_replay(self.current_replay_index + 1)

                # Previous frame
                elif event.key == pygame.K_UP:
                    self.current_frame = max(0, self.current_frame - 1)
                    self.paused = True

                # Next frame
                elif event.key == pygame.K_DOWN:
                    self.current_frame = min(
                        len(self.frames) - 1, self.current_frame + 1
                    )
                    self.paused = True

                # Restart replay
                elif event.key == pygame.K_r:
                    self.current_frame = 0
                    self.paused = False
                    self.last_frame_time = pygame.time.get_ticks()

                # Change playback speed
                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):
                    self.playback_speed_index = min(
                        len(PLAYBACK_SPEEDS) - 1,
                        self.playback_speed_index + 1,
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):
                    self.playback_speed_index = max(
                        0,
                        self.playback_speed_index - 1,
                    )

                # Jump to first frame
                elif event.key == pygame.K_HOME:
                    self.current_frame = 0
                    self.paused = True

                # Jump to last frame
                elif event.key == pygame.K_END:
                    self.current_frame = len(self.frames) - 1
                    self.paused = True

        return True

    def update_playback(self):
        """Advance to the next frame when the playback delay has passed."""

        if self.paused or not self.frames:
            return

        current_time = pygame.time.get_ticks()

        speed = PLAYBACK_SPEEDS[self.playback_speed_index]
        frame_delay = 1000 / speed

        if current_time - self.last_frame_time >= frame_delay:

            if self.current_frame < len(self.frames) - 1:
                self.current_frame += 1
            else:
                self.paused = True

            self.last_frame_time = current_time

    # ==========================
    # DRAWING HELPERS
    # ==========================

    def draw_grid(self):
        """Draw the chequered board grid."""

        for row in range(
            PLAY_RECT.top,
            PLAY_RECT.bottom,
            GRID,
        ):

            for col in range(
                PLAY_RECT.left,
                PLAY_RECT.right,
                GRID,
            ):

                if ((row // GRID + col // GRID) % 2) == 0:
                    colour = LIGHTGREEN
                else:
                    colour = DARKGREEN

                pygame.draw.rect(
                    self.screen,
                    colour,
                    (col, row, GRID, GRID),
                )

    def draw_image(self, image, position, fallback_colour):
        """Draw an image, or a coloured square if the image is missing."""

        if image is not None:
            self.screen.blit(image, position)

        else:
            pygame.draw.rect(
                self.screen,
                fallback_colour,
                (position[0], position[1], GRID, GRID),
            )

    def draw_head(self, frame):
        """Draw the snake head for the current frame."""

        direction = frame["direction"]

        dx, dy = direction

        if dx > 0:
            image = self.head["right"]
        elif dx < 0:
            image = self.head["left"]
        elif dy > 0:
            image = self.head["down"]
        else:
            image = self.head["up"]

        self.draw_image(
            image,
            frame["head"],
            GREEN,
        )

    def draw_body(self, frame):
        """Draw the snake body and tail for the current frame."""

        head = tuple(frame["head"])
        body = [tuple(position) for position in frame["body"]]

        segments = [head] + body

        # Draw body segments
        for i in range(1, len(segments) - 1):

            prev = segments[i - 1]
            cur = segments[i]
            nxt = segments[i + 1]

            px, py = prev
            cx, cy = cur
            nx, ny = nxt

            if px == nx:
                image = self.body["vertical"]

            elif py == ny:
                image = self.body["horizontal"]

            elif (px < cx and ny < cy) or (nx < cx and py < cy):
                image = self.body["tl"]

            elif (px > cx and ny < cy) or (nx > cx and py < cy):
                image = self.body["tr"]

            elif (px < cx and ny > cy) or (nx < cx and py > cy):
                image = self.body["bl"]

            else:
                image = self.body["br"]

            self.draw_image(image, cur, DARKERGREEN)

        # Draw tail
        if body:

            tail = body[-1]

            before = body[-2] if len(body) > 1 else head

            tx, ty = tail
            bx, by = before

            if bx > tx:
                image = self.tail["left"]

            elif bx < tx:
                image = self.tail["right"]

            elif by > ty:
                image = self.tail["up"]

            else:
                image = self.tail["down"]

            self.draw_image(image, tail, DARKERGREEN)

    def draw_food(self, frame):
        """Draw the food for the current frame."""

        food = frame.get("food")

        if food is None:
            return

        self.draw_image(
            self.apple,
            food,
            RED,
        )

    def draw_attempted_head(self, frame):
        """Highlight the cell the snake tried to move into."""

        attempted_head = frame.get("attempted_head")

        if attempted_head is None:
            return

        x, y = attempted_head

        # Highlight the position where the snake attempted to move
        pygame.draw.rect(
            self.screen,
            RED,
            (x + 3, y + 3, GRID - 6, GRID - 6),
            4,
        )

        # Cross marker
        pygame.draw.line(
            self.screen,
            RED,
            (x + 10, y + 10),
            (x + GRID - 10, y + GRID - 10),
            3,
        )

        pygame.draw.line(
            self.screen,
            RED,
            (x + GRID - 10, y + 10),
            (x + 10, y + GRID - 10),
            3,
        )

    # ==========================
    # INFORMATION PANEL
    # ==========================

    def draw_text(self, text, x, y, colour=BLACK, font=None):
        """Draw one line of text at the given position."""

        if font is None:
            font = self.small_font

        surface = font.render(
            str(text),
            True,
            colour,
        )

        self.screen.blit(surface, (x, y))

    def draw_information(self, frame):
        """Draw the game information and keyboard instructions."""
        
        if not self.replay_data:
            return

        game_number = self.replay_data.get("game_number", "?")
        score = frame.get("score", 0)
        reward = frame.get("reward", 0)
        action = frame.get("action")
        done = frame.get("done", False)
        death_reason = frame.get("death_reason")

        speed = PLAYBACK_SPEEDS[self.playback_speed_index]

        # Title
        title = self.title_font.render(
            "Snake RL Replay Viewer",
            True,
            DARKERGREEN,
        )

        title_rect = title.get_rect(center=(WINDOW_SIZE // 2, 30))

        self.screen.blit(title, title_rect)

        # Game information
        self.draw_text(
            f"Game: {game_number}",
            10,
            58,
        )

        self.draw_text(
            f"Score: {score}",
            150,
            58,
        )

        self.draw_text(
            f"Frame: {self.current_frame + 1}/{len(self.frames)}",
            270,
            58,
        )

        status = "PAUSED" if self.paused else "PLAYING"

        self.draw_text(
            status,
            470,
            58,
            RED if self.paused else GREEN,
        )

        # Controls and action information
        action_names = {
            0: "Straight",
            1: "Left",
            2: "Right",
            None: "Initial frame",
        }

        self.draw_text(
            f"Action: {action_names.get(action, 'Unknown')}",
            10,
            612,
        )

        self.draw_text(
            f"Reward: {reward:.2f}",
            180,
            612,
        )

        self.draw_text(
            f"Speed: {speed} FPS",
            330,
            612,
        )

        # Death information
        if done:

            if death_reason:
                message = f"Game Over: {death_reason}"

            else:
                message = "Game Completed!"

            self.draw_text(
                message,
                10,
                635,
                RED,
            )

        # Keyboard instructions
        self.draw_text(
            "SPACE: Pause/Resume",
            10,
            660,
        )

        self.draw_text(
            "LEFT/RIGHT: Change Replay",
            190,
            660,
        )

        self.draw_text(
            "UP/DOWN: Step Frame",
            420,
            660,
        )

    # ==========================
    # RENDER
    # ==========================

    def render(self):
        """Draw the current replay frame to the screen."""

        self.screen.fill(CREAM)

        if not self.frames:

            message = self.title_font.render(
                "No replay files found",
                True,
                RED,
            )

            rect = message.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))

            self.screen.blit(message, rect)

            self.draw_text(
                "Place replay JSON files inside the replays folder.",
                120,
                380,
            )

            pygame.display.flip()
            return

        if not self.replay_data:
            return

        frame = self.frames[self.current_frame]

        # Board
        pygame.draw.rect(
            self.screen,
            DARKERGREEN,
            PLAY_RECT,
            4,
        )

        self.draw_grid()

        # Game objects
        self.draw_food(frame)
        self.draw_body(frame)
        self.draw_head(frame)

        # Highlight collision destination
        self.draw_attempted_head(frame)

        # Information
        self.draw_information(frame)

        pygame.display.flip()

    # ==========================
    # MAIN LOOP
    # ==========================

    def run(self):
        """Run the replay viewer until the window is closed."""
        
        running = True

        while running:

            running = self.handle_input()

            self.update_playback()

            self.render()

            self.clock.tick(FPS)

        pygame.quit()


# ==========================
# RUN
# ==========================

if __name__ == "__main__":

    viewer = ReplayViewer()
    viewer.run()
