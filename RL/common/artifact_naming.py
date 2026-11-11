"""Shared naming helpers for Snake RL training artifacts."""

import re


def experiment_file_prefix(experiment_name: str) -> str:
    """Return the standard filename prefix for an RL experiment."""
    
    normalized_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(experiment_name).strip().lower(),
    ).strip("_")

    if not normalized_name:
        raise ValueError("Experiment name must contain letters or numbers.")

    return f"snake_{normalized_name}_RL"
