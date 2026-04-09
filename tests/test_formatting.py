"""Tests for evaluation output formatting."""
from __future__ import annotations

import unittest

from mario_rl.utils.formatting import format_evaluation_summary, format_percentage
from mario_rl.utils.metrics import EvaluationSummary


class FormattingTests(unittest.TestCase):
    def test_percentage_formatting(self) -> None:
        self.assertEqual(format_percentage(0.4), "40%")

    def test_summary_formatting_contains_expected_lines(self) -> None:
        summary = EvaluationSummary(
            episodes=3,
            avg_reward=12.5,
            best_reward=20.0,
            avg_distance=110.0,
            best_distance=150,
            clear_rate=1 / 3,
        )
        rendered = format_evaluation_summary(summary)
        self.assertIn("episodes: 3", rendered)
        self.assertIn("clear_rate: 33%", rendered)


if __name__ == "__main__":
    unittest.main()
