"""Tests for shared runtime configuration."""
from __future__ import annotations

import unittest

from mario_rl.config import DEFAULT_CONFIG
from mario_rl.env.mario_env import MarioEnv


class ConfigTests(unittest.TestCase):
    def test_default_config_matches_environment_shape(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.state_dim, MarioEnv.STATE_DIM)

    def test_default_checkpoint_paths_are_stable(self) -> None:
        self.assertTrue(DEFAULT_CONFIG.latest_checkpoint.endswith("latest.pt"))
        self.assertTrue(DEFAULT_CONFIG.human_checkpoint.endswith("human_trained.pt"))


if __name__ == "__main__":
    unittest.main()
