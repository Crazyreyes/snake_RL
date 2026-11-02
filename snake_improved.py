import pygame, random

# Initialize pygame
pygame.init()

# Draw the background grid
def draw_grid(surface):
    for row in range(PLAY_RECT.top, PLAY_RECT.bottom, SNAKE_SIZE):
        for col in range(PLAY_RECT.left, PLAY_RECT.right, SNAKE_SIZE):
            if ((row // SNAKE_SIZE + col // SNAKE_SIZE) % 2) == 0:
                colour = LIGHTGREEN
            else:
                colour = DARKGREEN

            pygame.draw.rect(surface, colour, (col, row, SNAKE_SIZE, SNAKE_SIZE))

# Determines the correct image to be used for the next snake body
def get_body_image(prev, curr, next):
    px, py, _, _ = prev
    cx, cy, _, _ = curr
    nx, ny, _, _ = next

    dx1, dy1 = px - cx, py - cy
    dx2, dy2 = nx - cx, ny - cy

    # Horizontal
    if dy1 == 0 and dy2 == 0:
        return body_horizontal

    # Vertical
    if dx1 == 0 and dx2 == 0:
        return body_vertical

    # Corners
    if (dx1 < 0 and dy2 < 0) or (dx2 < 0 and dy1 < 0):
        return body_topleft

    if (dx1 > 0 and dy2 < 0) or (dx2 > 0 and dy1 < 0):
        return body_topright

    if (dx1 < 0 and dy2 > 0) or (dx2 < 0 and dy1 > 0):
        return body_bottomleft

    if (dx1 > 0 and dy2 > 0) or (dx2 > 0 and dy1 > 0):
        return body_bottomright

# Determins the correct tail image to be used 
def get_tail_image(tail, before_tail):
    tx, ty, _, _ = tail
    bx, by, _, _ = before_tail

    dx = bx - tx
    dy = by - ty

    if dx > 0:
        return tail_left    # body is right → tail points left
    if dx < 0:
        return tail_right   # body is left → tail points right
    if dy > 0:
        return tail_up      # body is below → tail points up
    if dy < 0:
        return tail_down    # body is above → tail points down

# Spawn apples only on blank grids
def spawn_apple():
    while True:
        x = random.randrange(PLAY_RECT.left, PLAY_RECT.right, SNAKE_SIZE)
        y = random.randrange(PLAY_RECT.top, PLAY_RECT.bottom, SNAKE_SIZE)
        coord = (x, y, SNAKE_SIZE, SNAKE_SIZE)

        if coord != head_coord and coord not in body_coords:
            return (x, y)


# Set display surface
WINDOW_WIDTH = 680 #880
WINDOW_HEIGHT = 680 #880
display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("~~Snake~~")

# Set FPS and clock
FPS = 60
clock = pygame.time.Clock()

MOVE_DELAY = 5   # move once every 5 frames
frame_count = 0

# Set game values
SNAKE_SIZE = 40
MARGIN = 80

PLAY_RECT = pygame.Rect(MARGIN, MARGIN, WINDOW_WIDTH - MARGIN * 2, WINDOW_HEIGHT - MARGIN * 2)

head_x = round((WINDOW_WIDTH // 2) / SNAKE_SIZE) * SNAKE_SIZE
head_y = round((WINDOW_HEIGHT // 2) / SNAKE_SIZE) * SNAKE_SIZE

snake_dx = 0
snake_dy = 0

score = 0

# Set colours
LIGHTGREEN = (170, 215, 81)
GREEN = (0, 255, 0)
DARKGREEN = (167, 209, 61)
DARKERGREEN = (10, 50, 10)
RED = (255, 0, 0)
DARKRED = (150, 0, 0)
CREAM = (255, 253, 208)

# Set fonts
font = pygame.font.SysFont("gavriola", 48)

# Set text
title_text = font.render("~~Snake~~", True, GREEN)
title_rect = title_text.get_rect()
title_rect.topleft = (WINDOW_WIDTH // 2, 10)
title_rect.top = 10

score_text = font.render("Score: " + str(score), True, GREEN)
score_rect = score_text.get_rect()
score_rect.topleft = (10, 10)

game_over_text = font.render("GAMEOVER", True, RED, DARKERGREEN)
game_over_rect = game_over_text.get_rect()
game_over_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

continue_text = font.render("Press any key to play again", True, GREEN, DARKRED)
continue_rect = continue_text.get_rect()
continue_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 64)

# Set sounds and music
pick_up_sound = pygame.mixer.Sound("./assets/pick_up_sound.wav")

# Load images
head_up = pygame.image.load("./improved_assets/head_up.png")
head_down = pygame.image.load("./improved_assets/head_down.png")
head_left = pygame.image.load("./improved_assets/head_left.png")
head_right = pygame.image.load("./improved_assets/head_right.png")

body_horizontal = pygame.image.load("./improved_assets/body_horizontal.png")
body_vertical = pygame.image.load("./improved_assets/body_vertical.png")
body_topleft = pygame.image.load("./improved_assets/body_topleft.png")
body_topright = pygame.image.load("./improved_assets/body_topright.png")
body_bottomleft = pygame.image.load("./improved_assets/body_bottomleft.png")
body_bottomright = pygame.image.load("./improved_assets/body_bottomright.png")

tail_up = pygame.image.load("./improved_assets/tail_up.png")
tail_down = pygame.image.load("./improved_assets/tail_down.png")
tail_left = pygame.image.load("./improved_assets/tail_left.png")
tail_right = pygame.image.load("./improved_assets/tail_right.png")

apple_image = pygame.image.load("./improved_assets/apple.png")


# Set images (in this case, using simple rects so just create their coordinates)
# For a rect, you need (top-left x, top-left y, width, height)
head_coord = (head_x, head_y, SNAKE_SIZE, SNAKE_SIZE)
head_rect = pygame.draw.rect(display_surface, GREEN, head_coord)
head_image = head_right # Default head sprite

body_coords = []

apple_coord = spawn_apple()
apple_x, apple_y = apple_coord
apple_rect = pygame.Rect(apple_x, apple_y, SNAKE_SIZE, SNAKE_SIZE)

# The main game loop
running = True
while running:
    # Check to see if the user wants to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Move the snake
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                if snake_dx == SNAKE_SIZE and snake_dy == 0:
                    snake_dx = SNAKE_SIZE
                    snake_dy = 0
                else:
                    snake_dx = -1 * SNAKE_SIZE
                    snake_dy = 0
            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                if snake_dx == -1 * SNAKE_SIZE and snake_dy == 0:
                    snake_dx = -1 * SNAKE_SIZE
                    snake_dy = 0
                else:
                    snake_dx = SNAKE_SIZE
                    snake_dy = 0
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                if snake_dx == 0 and snake_dy == SNAKE_SIZE:
                    snake_dx = 0
                    snake_dy = SNAKE_SIZE
                else:
                    snake_dx = 0
                    snake_dy = -1 * SNAKE_SIZE
            if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                if snake_dx == 0 and snake_dy == -1 * SNAKE_SIZE:
                    snake_dx = 0
                    snake_dy = -1 * SNAKE_SIZE
                else:
                    snake_dx = 0
                    snake_dy = SNAKE_SIZE

    # Check if it is the 20th frame and update movement
    frame_count += 1
    if frame_count % MOVE_DELAY == 0:
        # Add the head coordinate to the first index of the body coordinate list
        # This will essentially move all of the snakes body by one position in the list
        body_coords.insert(0, head_coord)
        if len(body_coords) > score:
            body_coords.pop()

        # Update the x, y positions of the snakes head and make a new coordinate
        head_x += snake_dx
        head_y += snake_dy
        head_coord = (head_x, head_y, SNAKE_SIZE, SNAKE_SIZE)

    # Check for game over
    if not PLAY_RECT.contains(head_rect) or head_coord in body_coords:
        display_surface.blit(game_over_text, game_over_rect)
        display_surface.blit(continue_text, continue_rect)
        pygame.display.update()

        # Pause the game until the player presses a key, then reset the game
        is_paused = True
        while is_paused:
            for event in pygame.event.get():
                # The player wants to play again
                if event.type == pygame.KEYDOWN:
                    score = 0

                    head_x = round((WINDOW_WIDTH // 2) / SNAKE_SIZE) * SNAKE_SIZE
                    head_y = round((WINDOW_HEIGHT // 2) / SNAKE_SIZE) * SNAKE_SIZE
                    head_coord = (head_x, head_y, SNAKE_SIZE, SNAKE_SIZE)

                    apple_coord = spawn_apple()
                    apple_x, apple_y = apple_coord 

                    body_coords = []

                    snake_dx = 0
                    snake_dy = 0

                    is_paused = False

                # The player wants to quit
                if event.type == pygame.QUIT:
                    is_paused = False
                    running = False

    # Check for collisions
    if head_rect.colliderect(apple_rect):
        score += 1
        pick_up_sound.play()

        apple_coord = spawn_apple()
        apple_x, apple_y = apple_coord 

    if snake_dx > 0:
        head_image = head_right
    elif snake_dx < 0:
        head_image = head_left
    elif snake_dy > 0:
        head_image = head_down
    elif snake_dy < 0:
        head_image = head_up

    # Update HUD
    score_text = font.render("Score: " + str(score), True, GREEN)

    # Fill the surface
    display_surface.fill(CREAM)
    pygame.draw.rect(display_surface, DARKERGREEN, PLAY_RECT, 4)

    draw_grid(display_surface)

    # Blit HUD
    display_surface.blit(title_text, title_rect)
    display_surface.blit(score_text, score_rect)

    # Blit assets
    display_surface.blit(head_image, (head_x, head_y))
    head_rect = pygame.Rect(head_x, head_y, SNAKE_SIZE, SNAKE_SIZE)

    display_surface.blit(apple_image, (apple_x, apple_y))
    apple_rect = pygame.Rect(apple_x, apple_y, SNAKE_SIZE, SNAKE_SIZE)

    snake_segments = [head_coord] + body_coords
    for i in range(1, len(snake_segments) - 1):
        prev = snake_segments[i - 1]
        curr = snake_segments[i]
        next = snake_segments[i + 1]

        body_img = get_body_image(prev, curr, next)
        x, y, _, _ = curr
        display_surface.blit(body_img, (x, y))
    
    if len(snake_segments) >= 2:
        tail = snake_segments[-1]
        before_tail = snake_segments[-2]

        tail_img = get_tail_image(tail, before_tail)
        x, y, _, _ = tail
        display_surface.blit(tail_img, (x, y))
    # Update display and tick clock
    pygame.display.update()
    clock.tick(FPS)
# End the game
pygame.quit()
