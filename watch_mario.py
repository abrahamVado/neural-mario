"""Watch Mario Play with Real-Time Web Visualization!"""
import os
import torch
import time
import json
import asyncio
import threading
import numpy as np
import websockets
from mario_rl.env.mario_env import MarioEnv
from mario_rl.brain.dqn_brain import SimpleDQNAgent

# Global state for the visualization
latest_data = {
    "layers": [],
    "outputs": [],
    "decision": 0,
    "inputs": []
}
lock = threading.Lock()

def downsample(arr, target_n):
    """Downsample a numpy array to target_n elements."""
    if len(arr) <= target_n:
        return arr.tolist()
    # Simple sampling
    indices = np.linspace(0, len(arr) - 1, target_n, dtype=int)
    return arr[indices].tolist()

async def ws_handler(websocket):
    print("New client connected!")
    try:
        while True:
            with lock:
                data = latest_data
            if data['inputs']: # Only send if we have data
                await websocket.send(json.dumps(data))
            await asyncio.sleep(0.04) # ~25 FPS updates
    except websockets.exceptions.ConnectionClosed:
        pass

async def start_server():
    print("🚀 WebSocket server starting on ws://localhost:8765")
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future() # Run forever

def run_server_thread():
    asyncio.run(start_server())

def watch():
    global latest_data
    
    # Start WS Server
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    server_thread.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👀 Watching Mario on {device}")
    
    # Setup
    env = MarioEnv(world=1, stage=1, apply_cheats=False)
    # Using the dimensions from the file directly
    agent = SimpleDQNAgent(state_dim=204, action_dim=7, device=device)
    
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
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                
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
            
            # Prepare data for WS
            # Mapping: input -> hidden1 -> hidden2 -> output
            # We downsample hidden layers for visual clarity
            vis_input = downsample(activations['input'], 20)
            vis_h1 = downsample(activations['hidden1'], 25)
            vis_h2 = downsample(activations['hidden2'], 25)
            vis_out = activations['output'].tolist()
            
            with lock:
                latest_data = {
                    "inputs": vis_input,
                    "layers": [vis_h1, vis_h2], # Middle layers
                    "outputs": vis_out,
                    "decision": action
                }

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
