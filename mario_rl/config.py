"""Shared configuration for Mario RL scripts."""
from __future__ import annotations

from dataclasses import dataclass

from mario_rl.env.mario_env import MarioEnv


@dataclass(frozen=True)
class MarioRunConfig:
    """Shared runtime configuration for training and evaluation."""

    world: int = 1
    stage: int = 1
    action_dim: int = 7
    state_dim: int = MarioEnv.STATE_DIM
    max_steps: int = 5000
    apply_cheats: bool = False
    checkpoint_dir: str = "checkpoints"
    latest_checkpoint: str = "checkpoints/latest.pt"
    human_checkpoint: str = "checkpoints/human_trained.pt"
    archive_dir: str = "checkpoints/old_v1"
    save_interval: int = 50
    num_episodes: int = 5000


DEFAULT_CONFIG = MarioRunConfig()
