"""Mario RL Q-Network architecture."""
from __future__ import annotations
import torch
import torch.nn as nn


class MarioQNetwork(nn.Module):
    """Q-Network for Mario with Dueling Architecture.
    
    Input: 25 features from Mario environment
    Hidden: 256 → 256 (Shared)
    Value Stream: 256 → 1
    Advantage Stream: 256 → Action Dim
    Output: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    
    def __init__(self, state_dim: int = 25, action_dim: int = 7):
        super().__init__()
        
        # Shared Feature Layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Value Stream
        self.value_fc = nn.Linear(256, 1)
        
        # Advantage Stream
        self.advantage_fc = nn.Linear(256, action_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Dueling Network."""
        # Shared features
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Streams
        val = self.value_fc(x)
        adv = self.advantage_fc(x)
        
        # Combine: Q = V + (A - mean(A))
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        
        return q
    
    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass that also returns intermediate activations for visualization."""
        activations = {}
        
        # Input layer
        activations['input'] = x.detach().cpu().numpy().flatten()
        
        # Hidden layer 1
        h1 = self.fc1(x)
        activations['hidden1'] = h1.detach().cpu().numpy().flatten()
        a1 = self.relu(h1)
        
        # Hidden layer 2 (Shared)
        h2 = self.fc2(a1)
        activations['hidden2'] = h2.detach().cpu().numpy().flatten()
        a2 = self.relu(h2)
        
        # Streams
        val = self.value_fc(a2)
        adv = self.advantage_fc(a2)
        
        activations['value'] = val.detach().cpu().numpy().flatten()
        activations['advantage'] = adv.detach().cpu().numpy().flatten()
        
        # Combine
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        activations['output'] = q.detach().cpu().numpy().flatten()
        
        return q, activations
