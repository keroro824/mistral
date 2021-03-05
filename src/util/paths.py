"""
paths.py

Utility function for initializing the appropriate directories/sub-directories on the start of each run. Decoupled from
main code in case we want separate directory structures/artifact storage based on infrastructure (e.g., NLP Cluster vs.
GCP).
"""
import os
from typing import Dict

from .registry import REGISTRY


def create_paths(run_id: str, model: str, run_dir: str, cache_dir: str) -> Dict[str, str]:
    """
    Create the necessary directories and sub-directories conditioned on the `run_id`, checkpoint directory, and cache
    directory.

    :param run_id: Unique Run Identifier.
    :param model: Huggingface.Transformers Model ID for specifying the desired configuration.
    :param run_dir: Path to run directory to save model checkpoints and run metrics.
    :param cache_dir: Path to artifacts/cache directory to store any intermediate values, configurations, etc.

    :return: Dictionary mapping str ids --> paths on the filesystem.
    """
    paths = {
        # Top-Level Checkpoint Directory for Given Run
        "runs": os.path.join(run_dir, run_id),
        # Logging Directory (HF defaults to Tensorboard -- TODO 19 :: Remove Tensorboard and just use W&B and Custom?
        "logs": os.path.join(run_dir, run_id, "logs"),
        # WandB Save Directory
        "wandb": os.path.join(run_dir, run_id, "wandb"),
        # Cache Directories for various components
        "configs": os.path.join(cache_dir, f"{REGISTRY[model]}-configs"),
        "tokenizer": os.path.join(cache_dir, f"{REGISTRY[model]}-tokenizer"),
        "dataset": os.path.join(cache_dir, f"{REGISTRY[model]}-dataset"),
    }

    # Programatically Create Paths for each Directory
    for p in paths:
        os.makedirs(paths[p], exist_ok=True)

    return paths
