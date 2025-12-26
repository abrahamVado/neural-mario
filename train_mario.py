"""Train DQN agent to play Super Mario Bros - Self-contained version."""
from __future__ import annotations
import os
import torch
from tqdm import tqdm

from mario_rl.env.mario_env import MarioEnv
from mario_rl.agents.networks import MarioQNetwork
from mario_rl.brain.dqn_brain import SimpleDQNAgent
from mario_rl.utils.server import start_background_server, update_visualization
import logging
import numpy as np
import random


def train():
    """Train Mario DQN agent."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🍄 Starting Mario RL Training!")
    print(f"Device: {device}")
    print(f"Network: 204 (Grid+Helpers) → 256 → 256 → 7")
    print("-" * 50)

    # Setup logging
    logging.basicConfig(
        filename='training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    logging.info(f"🍄 Starting Mario RL Training on {device}")
    
    # Start Visualization Server
    start_background_server()

    
    # Create environment and agent
    # Enable 'apply_cheats' to give Mario infinite Fire Flower! 🍄🔥
    env = MarioEnv(world=1, stage=1, max_steps=5000, apply_cheats=False)
    
    # State Dim = 204 (Grid features + Long Jump helpers)
    agent = SimpleDQNAgent(state_dim=204, action_dim=7, device=device)
    
    # Training
    num_episodes = 5000
    save_interval = 50
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # CHECKPOINT COMPATIBILITY CHECK
    # Because we changed the architecture/input size, old checkpoints will fail.
    # We should detect if we need to start fresh or archive them.
    
    if os.path.exists("checkpoints/latest.pt"):
        try:
            # Try to load just to check
            dummy = torch.load("checkpoints/latest.pt")
            # If shape mismatch, it might not fail until load_state_dict, 
            # but let's be safe: if user said "Go ahead" knowing it's a restart,
            # we should archive the old ones to avoid confusion.
            
            # But wait, if we ARE resuming a run we just started with the new code, 
            # we don't want to archive it!
            # Let's assume if it fails to load, THEN we archive.
            pass
        except:
            pass
            
    # Resume Training: Load Latest Model if exists
    if os.path.exists("checkpoints/latest.pt"):
        print("\n🔄 Checking for existing checkpoint...")
        try:
            agent.load("checkpoints/latest.pt")
            print(f"✅ Loaded checkpoint! Resuming from step {agent.steps}")
        except Exception as e:
            print(f"⚠️ Checkpoint incompatible (likely old architecture): {e}")
            print("📦 Archiving old checkpoints to 'checkpoints/old_v1'...")
            os.makedirs("checkpoints/old_v1", exist_ok=True)
            if os.path.exists("checkpoints/latest.pt"):
                os.rename("checkpoints/latest.pt", "checkpoints/old_v1/latest.pt")
            if os.path.exists("checkpoints/human_trained.pt"):
                 # We keep human trained but maybe rename it if we can't use it?
                 # Actually, we can't use human_trained.pt either if inputs changed!
                 print("⚠️ 'human_trained.pt' is also incompatible. Archiving.")
                 os.rename("checkpoints/human_trained.pt", "checkpoints/old_v1/human_trained.pt")
            
            print("🚀 Starting FRESH training with NEW Vision System!")
            
    # Transfer Learning: Load Human Model if exists AND we are not resuming
    elif os.path.exists("checkpoints/human_trained.pt"):
        print("\n🎓 FOUND HUMAN-TRAINED MODEL! Importing Knowledge...")
        try:
            agent.load("checkpoints/human_trained.pt")
            print("✅ Transfer Learning ENABLED.")
            agent.epsilon_start = 0.15
        except Exception as e:
            print(f"⚠️ Human model incompatible: {e}")
            print("🚀 Starting FRESH training.")
    
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):


        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        total_loss = 0.0
        total_q = 0.0
        updates = 0
        
        while not done:
            # action = agent.act(state)
            
            # --- Visualization & Action Selection ---
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values, activations = agent.q_net.forward_with_activations(state_t)
                
                # Epsilon Greedy
                if random.random() < agent.epsilon():
                    action = random.randint(0, agent.action_dim - 1)
                else:
                    action = q_values.argmax().item()
            
            # Update Visualization (Broadcast)
            update_visualization(activations, action)
            # ----------------------------------------
            
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss, avg_q = agent.update()
            
            if loss > 0:
                total_loss += loss
                total_q += avg_q
                updates += 1
            
            total_reward += reward
            steps += 1
            state = next_state
        
        # Log progress
        if episode % 10 == 0:
            max_x = info.get('x_pos', 0)
            flag = info.get('flag_get', False)
            avg_loss = total_loss / updates if updates > 0 else 0.0
            avg_q_val = total_q / updates if updates > 0 else 0.0
            
            log_msg = (f"Ep {episode}: Reward={total_reward:.1f}, Steps={steps}, "
                       f"MaxX={max_x}, Flag={'✅' if flag else '❌'}, ε={agent.epsilon():.3f}, "
                       f"Loss={avg_loss:.4f}, Q={avg_q_val:.2f}")
            tqdm.write(log_msg)
            logging.info(log_msg)

        
        # Save checkpoint
        if episode % save_interval == 0:
            agent.save(f"checkpoints/mario_ep{episode}.pt")
            agent.save("checkpoints/latest.pt")
            
    print("\n✅ Training complete!")
    env.close()


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.error("❌ Training Crashed!", exc_info=True)
        print(f"\n❌ Training Crashed! Check training.log for details.\n{e}")

