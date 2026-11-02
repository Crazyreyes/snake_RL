import pygame

from agents.dqn_agent import DQNAgent
from common.config import RENDER
from common.logger import log_game, setup_logger
from common.replay import ReplayRecorder
from common.renderer import SnakeRenderer
from common.snake_game import SnakeGame


def run_training(
    reward_function,
    name,
    double_dqn=False,
    epsilon_start=80,
    epsilon_decay=0.5,
    epsilon_min=5,
    replay_threshold=30,
):
    logger, csv_path = setup_logger(name)
    agent = DQNAgent(double_dqn, epsilon_start, epsilon_decay, epsilon_min)
    game = SnakeGame()
    renderer = SnakeRenderer(f"Snake RL - {name}")
    replay = ReplayRecorder()
    replay.reset(game)

    steps = 0
    total_reward = 0.0
    running = True

    while running:
        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        if not running:
            break

        state = game.get_state()
        action = agent.get_action(state)
        reward, done, score, info = game.step(action, reward_function)
        next_state = game.get_state()

        replay.record(
            game,
            action,
            reward,
            done,
            info["death_reason"],
            info["new_head"] if info["death_reason"] else None,
        )

        agent.remember(state, action, reward, next_state, done)
        agent.train_long()

        steps += 1
        total_reward += reward

        if done:
            agent.n_games += 1

            if agent.n_games % agent.update_target_every == 0:
                agent.update_target_network()

            saved = score >= replay_threshold
            if saved:
                path = replay.save(agent.n_games, score, name)
                logger.info("Replay saved: %s", path.name)
                print("Replay saved:", path.name)

            log_game(
                logger,
                csv_path,
                agent.n_games,
                score,
                agent.epsilon,
                steps,
                total_reward,
                saved,
            )

            print(
                f"Game {agent.n_games} | Score {score} | "
                f"Epsilon {agent.epsilon:.2f} | Steps {steps}"
            )

            game.reset()
            replay.reset(game)
            steps = 0
            total_reward = 0.0

        renderer.render(game)

    if RENDER:
        pygame.quit()
