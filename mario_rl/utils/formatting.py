"""Formatting helpers for CLI output."""
from __future__ import annotations

from mario_rl.utils.metrics import EvaluationSummary


def format_percentage(value: float) -> str:
    """Format a ratio as a whole-number percentage."""
    return f"{value:.0%}"


def format_evaluation_summary(summary: EvaluationSummary) -> str:
    """Render evaluation metrics in a readable multi-line block."""
    lines = [
        "Mario checkpoint evaluation",
        f"episodes: {summary.episodes}",
        f"avg_reward: {summary.avg_reward:.2f}",
        f"best_reward: {summary.best_reward:.2f}",
        f"avg_distance: {summary.avg_distance:.2f}",
        f"best_distance: {summary.best_distance}",
        f"clear_rate: {format_percentage(summary.clear_rate)}",
    ]
    return "\n".join(lines)
