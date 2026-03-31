"""Evaluate saved Mario checkpoints over multiple episodes."""
from __future__ import annotations

import statistics

import torch

from mario_rl.brain.dqn_brain import SimpleDQNAgent
from mario_rl.config import DEFAULT_CONFIG
from mario_rl.env.mario_env import MarioEnv


def evaluate(num_episodes: int = 5) -> dict[str, float]:
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

    return {
        "episodes": float(num_episodes),
        "avg_reward": statistics.fmean(rewards) if rewards else 0.0,
        "best_reward": max(rewards) if rewards else 0.0,
        "avg_distance": statistics.fmean(distances) if distances else 0.0,
        "best_distance": max(distances) if distances else 0.0,
        "clear_rate": flags_cleared / num_episodes if num_episodes else 0.0,
    }


if __name__ == "__main__":
    summary = evaluate()
    print("Mario checkpoint evaluation")
    for key, value in summary.items():
        if key in {"episodes", "best_distance"}:
            print(f"{key}: {int(value)}")
        elif key == "clear_rate":
            print(f"{key}: {value:.0%}")
        else:
            print(f"{key}: {value:.2f}")
