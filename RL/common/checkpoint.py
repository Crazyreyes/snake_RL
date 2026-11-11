"""Saving and loading Snake RL checkpoints."""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

import os
import time
import tempfile

from common.artifact_naming import experiment_file_prefix
from common.config import CHECKPOINT_DIR, MAX_MEMORY

CHECKPOINT_VERSION = 1


def get_checkpoint_paths(experiment_name):
    """Return paths for training and inference checkpoints."""

    experiment_dir = CHECKPOINT_DIR / experiment_file_prefix(experiment_name)

    return (
        experiment_dir / "latest_training_checkpoint.pth",
        experiment_dir / "best_model.pth",
    )


def _atomic_torch_save(payload, path):
    """Save safely with retries for temporary Windows file locks."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        # Create a unique temporary file in the same directory.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
        ) as temporary_file:

            temporary_path = Path(temporary_file.name)

            torch.save(payload, temporary_file)

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # Retry if Windows temporarily locks the destination.
        for attempt in range(10):
            try:
                os.replace(temporary_path, path)
                return

            except PermissionError:
                if attempt == 9:
                    raise

                print(
                    f"Checkpoint file temporarily locked. "
                    f"Retrying ({attempt + 1}/10)..."
                )

                time.sleep(0.5)

    finally:
        # Remove temporary files if saving fails.
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _torch_load(path, map_location):
    """Load a trusted checkpoint created by this project."""

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=map_location)


def save_training_checkpoint(
    agent,
    path,
    experiment_name,
    best_score,
):
    """Save the complete training state."""

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "training",
        "experiment_name": experiment_name,
        "n_games": agent.n_games,
        "epsilon": agent.epsilon,
        "best_score": best_score,
        "agent_config": {
            "double_dqn": agent.double_dqn,
            "epsilon_start": agent.epsilon_start,
            "epsilon_decay": agent.epsilon_decay,
            "epsilon_min": agent.epsilon_min,
            "update_target_every": agent.update_target_every,
            "loss": agent.loss,
        },
        "model_state_dict": agent.model.state_dict(),
        "target_model_state_dict": agent.target_model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "replay_memory": list(agent.memory),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }

    _atomic_torch_save(payload, path)


def load_training_checkpoint(
    agent,
    path,
    map_location="cpu",
):
    """Restore an agent from a training checkpoint."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {path}")

    payload = _torch_load(path, map_location)

    if payload.get("checkpoint_type") != "training":
        raise ValueError("This is not a training checkpoint.")

    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version.")

    saved_config = payload["agent_config"]
    
    saved_config = dict(saved_config)
    saved_config.setdefault("loss", agent.loss)

    current_config = {
        "double_dqn": agent.double_dqn,
        "epsilon_start": agent.epsilon_start,
        "epsilon_decay": agent.epsilon_decay,
        "epsilon_min": agent.epsilon_min,
        "update_target_every": agent.update_target_every,
        "loss": agent.loss,
    }

    if saved_config != current_config:
        raise ValueError(
            "Training settings do not match this checkpoint.\n"
            f"Saved: {saved_config}\n"
            f"Current: {current_config}"
        )

    # Restore neural networks.
    agent.model.load_state_dict(payload["model_state_dict"])

    agent.target_model.load_state_dict(payload["target_model_state_dict"])

    # Restore optimizer.
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])

    # Restore training progress.
    agent.n_games = int(payload["n_games"])
    agent.epsilon = float(payload["epsilon"])

    # Restore replay memory.
    agent.memory = deque(
        payload["replay_memory"],
        maxlen=MAX_MEMORY,
    )

    # Restore random number generator states.
    rng_state = payload["rng_state"]

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])

    torch.set_rng_state(rng_state["torch"].cpu())

    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])

    return {
        "n_games": agent.n_games,
        "epsilon": agent.epsilon,
        "best_score": int(payload.get("best_score", -1)),
        "experiment_name": payload.get("experiment_name"),
        "loss": payload.get("agent_config", {}).get("loss"),
    }


def save_best_model(
    agent,
    path,
    experiment_name,
    score,
):
    """Save the model weights needed for inference."""

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "inference",
        "experiment_name": experiment_name,
        "score": int(score),
        "loss": agent.loss,
        "model_config": {
            "input_size": 15,
            "hidden_size": 128,
            "output_size": 3,
        },
        "model_state_dict": agent.model.state_dict(),
    }

    _atomic_torch_save(payload, path)


def load_model_for_inference(
    path,
    map_location="cpu",
):
    """Load a trained model without creating a training agent."""

    from agents.dqn_agent import DQN

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    payload = _torch_load(path, map_location)

    if payload.get("checkpoint_type") != "inference":
        raise ValueError("This is not an inference checkpoint.")

    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version.")

    expected_config = {
        "input_size": 15,
        "hidden_size": 128,
        "output_size": 3,
    }

    if payload.get("model_config") != expected_config:
        raise ValueError(f"Unsupported architecture: {payload.get('model_config')}")

    model = DQN()

    model.load_state_dict(payload["model_state_dict"])

    model.eval()

    metadata = {
        "experiment_name": payload.get("experiment_name"),
        "score": int(payload.get("score", 0)),
        "loss": payload.get("loss", "unknown"),
    }

    return model, metadata
