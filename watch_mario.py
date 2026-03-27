"""Watch Mario Play with Real-Time Web Visualization!"""

import os
import time

import numpy as np
import torch

from mario_rl.brain.dqn_brain import SimpleDQNAgent
from mario_rl.env.mario_env import MarioEnv
from mario_rl.utils.server import start_background_server, update_visualization

def watch():
    # Start WS Server
    start_background_server()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👀 Watching Mario on {device}")
    
    # Setup
    env = MarioEnv(world=1, stage=1, apply_cheats=False)
    agent = SimpleDQNAgent(state_dim=MarioEnv.STATE_DIM, action_dim=7, device=device)
    
    # Load Model
    if os.path.exists("checkpoints/latest.pt"):
        print("✅ Loading latest model...")
        agent.load("checkpoints/latest.pt")
        # Turn off exploration to see what it truly learned
        agent.epsilon_start = 0.05 
        agent.epsilon_end = 0.05
    else:
        print("⚠️ No model found! Mario will play randomly.")
        
    print("🎮 Game Loop Started...")
    
    # Validation Loop
    while True:
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            env.render()
            
            # Get Action and Activations
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Check directly if using random (epsilon) or network
                # For visualization, we always want to run the network to see activations
                # even if we ultimately pick a random action.
                
                q_values, activations = agent.q_net.forward_with_activations(state_t)
                
                # Decision
                if np.random.random() < agent.epsilon_end:
                    action = np.random.randint(0, 7)
                    # We still use the network activations for viz
                else:
                    action = q_values.argmax().item()
            
            update_visualization(activations, action)
            
            # Step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            
            time.sleep(0.01) # Game speed control
            
        print(f"Game Over! Reward: {total_reward:.1f}")
        time.sleep(1.0)

if __name__ == "__main__":
    try:
        watch()
    except KeyboardInterrupt:
        print("\n👋 Bye!")
