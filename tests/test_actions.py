"""Tests for action naming helpers."""
from __future__ import annotations

import unittest

from mario_rl.utils.actions import action_name


class ActionNameTests(unittest.TestCase):
    def test_known_action_name(self) -> None:
        self.assertEqual(action_name(3), "right_run")

    def test_unknown_action_name(self) -> None:
        self.assertEqual(action_name(99), "unknown_99")


if __name__ == "__main__":
    unittest.main()
