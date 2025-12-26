"""DQN Brain for Mario."""
from __future__ import annotations
import torch
import numpy as np
from collections import deque
import random
from mario_rl.agents.networks import MarioQNetwork

class SimpleDQNAgent:
    """Simplified DQN agent for Mario."""
    
    def __init__(self, state_dim, action_dim=7, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        
        # Networks
        self.q_net = MarioQNetwork(state_dim, action_dim).to(device)
        self.target_net = MarioQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.0001)
        
        # Replay buffer
        self.memory = deque(maxlen=100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 128
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 100000
        self.steps = 0
        self.target_update_freq = 1000
        self.replay_initial = 1000
        
    def epsilon(self):
        """Current epsilon for exploration."""
        progress = min(self.steps / self.epsilon_decay, 1.0)
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
    
    def act(self, state):
        """Select action using epsilon-greedy."""
        if random.random() < self.epsilon():
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update Q-network using Double DQN."""
        if len(self.memory) < self.replay_initial or len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Log average Q-value for debugging
        avg_q = current_q.mean().item()
        
        # Double DQN Target
        with torch.no_grad():
            # Use Online Network to select best action for next state
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            # Use Target Network to evaluate that action
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and update
        loss = torch.nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item(), avg_q
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        
        # Robust loading for transfer learning (partial checkpoints)
        if 'target_net' in checkpoint:
            self.target_net.load_state_dict(checkpoint['target_net'])
        else:
            print("ℹ️ Transfer Learning: copying q_net to target_net")
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("ℹ️ Transfer Learning: Starting with fresh optimizer")
            
        self.steps = checkpoint.get('steps', 0)
