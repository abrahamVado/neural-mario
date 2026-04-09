"""Tests for evaluation metric helpers."""
from __future__ import annotations

import unittest

from mario_rl.utils.metrics import build_evaluation_summary


class MetricsTests(unittest.TestCase):
    def test_build_summary_with_values(self) -> None:
        summary = build_evaluation_summary([10.0, 20.0], [100, 140], 1)
        self.assertEqual(summary.episodes, 2)
        self.assertAlmostEqual(summary.avg_reward, 15.0)
        self.assertEqual(summary.best_distance, 140)
        self.assertAlmostEqual(summary.clear_rate, 0.5)

    def test_build_summary_with_empty_values(self) -> None:
        summary = build_evaluation_summary([], [], 0)
        self.assertEqual(summary.episodes, 0)
        self.assertEqual(summary.best_reward, 0.0)
        self.assertEqual(summary.best_distance, 0)


if __name__ == "__main__":
    unittest.main()
