
import pygame

from agents.dqn_agent import DQNAgent

from common.checkpoint import (
    get_checkpoint_paths,
    load_training_checkpoint,
    save_best_model,
    save_training_checkpoint,
)

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
    replay_threshold=40,
    resume=False,
    checkpoint_every=25,
):

    if checkpoint_every < 1:
        raise ValueError(
            "checkpoint_every must be at least 1."
        )

    logger, csv_path = setup_logger(name)

    agent = DQNAgent(
        double_dqn,
        epsilon_start,
        epsilon_decay,
        epsilon_min,
    )

    game = SnakeGame()

    renderer = SnakeRenderer(
        f"Snake RL - {name}"
    )

    replay = ReplayRecorder()
    replay.reset(game)

    latest_checkpoint_path, best_model_path = (
        get_checkpoint_paths(name)
    )

    best_score = -1

    # Resume previous training session.
    if resume:

        metadata = load_training_checkpoint(
            agent,
            latest_checkpoint_path,
        )

        best_score = metadata["best_score"]

        # Checkpoints are saved at episode boundaries.
        # Restore RNG states before generating the next game.
        game.reset()
        replay.reset(game)

        print(
            f"Resumed {name} training at game "
            f"{agent.n_games} with epsilon "
            f"{agent.epsilon:.2f}."
        )

        logger.info(
            "Resumed from checkpoint at game %s",
            agent.n_games,
        )

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

        reward, done, score, info = game.step(
            action,
            reward_function,
        )

        next_state = game.get_state()

        replay.record(
            game,
            action,
            reward,
            done,
            info["death_reason"],
            (
                info["new_head"]
                if info["death_reason"]
                else None
            ),
        )

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )

        agent.train_long()

        steps += 1
        total_reward += reward

        if done:

            agent.n_games += 1

            if agent.n_games % agent.update_target_every == 0:
                agent.update_target_network()

            saved = score >= replay_threshold

            if saved:

                path = replay.save(
                    agent.n_games,
                    score,
                    name,
                )

                logger.info(
                    "Replay saved: %s",
                    path.name,
                )

                print(
                    "Replay saved:",
                    path.name,
                )

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

            # Save a new best-performing inference model.
            if score > best_score:

                best_score = score

                save_best_model(
                    agent,
                    best_model_path,
                    name,
                    score,
                )

                print(
                    f"New best model saved: score {score}"
                )

            # Save resumable training checkpoint.
            if agent.n_games % checkpoint_every == 0:

                save_training_checkpoint(
                    agent,
                    latest_checkpoint_path,
                    name,
                    best_score,
                )

                print(
                    f"Training checkpoint saved at "
                    f"game {agent.n_games}: "
                    f"{latest_checkpoint_path}"
                )

            print(
                f"Game {agent.n_games} | "
                f"Score {score} | "
                f"Epsilon {agent.epsilon:.2f} | "
                f"Steps {steps}"
            )

            game.reset()
            replay.reset(game)

            steps = 0
            total_reward = 0.0

        renderer.render(game)

    if RENDER:
        pygame.quit()

    print(
        "Training stopped. Latest resumable checkpoint:",
        latest_checkpoint_path,
    )

    print(
        "Best inference model:",
        best_model_path,
    )
