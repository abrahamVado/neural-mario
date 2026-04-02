"""Evaluation metrics for Mario RL runs."""
from __future__ import annotations

from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics from evaluation episodes."""

    episodes: int
    avg_reward: float
    best_reward: float
    avg_distance: float
    best_distance: int
    clear_rate: float


def build_evaluation_summary(
    rewards: list[float], distances: list[int], flags_cleared: int
) -> EvaluationSummary:
    """Build a stable evaluation summary from per-episode values."""
    episodes = len(rewards)
    return EvaluationSummary(
        episodes=episodes,
        avg_reward=statistics.fmean(rewards) if rewards else 0.0,
        best_reward=max(rewards) if rewards else 0.0,
        avg_distance=statistics.fmean(distances) if distances else 0.0,
        best_distance=max(distances) if distances else 0,
        clear_rate=(flags_cleared / episodes) if episodes else 0.0,
    )
