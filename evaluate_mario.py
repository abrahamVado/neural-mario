"""Evaluate saved Mario checkpoints over multiple episodes."""
from __future__ import annotations

import argparse

import torch

from mario_rl.brain.dqn_brain import SimpleDQNAgent
from mario_rl.config import DEFAULT_CONFIG
from mario_rl.env.mario_env import MarioEnv
from mario_rl.utils.formatting import format_evaluation_summary
from mario_rl.utils.metrics import EvaluationSummary, build_evaluation_summary


def evaluate(num_episodes: int = 5) -> EvaluationSummary:
    """Run deterministic evaluation episodes and return summary stats."""
    config = DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MarioEnv(
        world=config.world,
        stage=config.stage,
        max_steps=config.max_steps,
        apply_cheats=config.apply_cheats,
    )
    agent = SimpleDQNAgent(state_dim=config.state_dim, action_dim=config.action_dim, device=device)
    agent.load(config.latest_checkpoint)
    agent.epsilon_start = 0.0
    agent.epsilon_end = 0.0

    rewards: list[float] = []
    distances: list[int] = []
    flags_cleared = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        max_x = 0
        flag = False

        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            max_x = max(max_x, int(info.get("x_pos", 0)))
            flag = flag or bool(info.get("flag_get", False))

        rewards.append(total_reward)
        distances.append(max_x)
        flags_cleared += int(flag)

    env.close()

    return build_evaluation_summary(rewards, distances, flags_cleared)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a Mario checkpoint.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes to run.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = evaluate(num_episodes=args.episodes)
    print(format_evaluation_summary(summary))
