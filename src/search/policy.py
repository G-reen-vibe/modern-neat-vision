"""Learned policy network for greedy complexification.

The policy is a small MLP that takes the current graph's feature vector
and outputs a probability distribution over growth operations.

Training: REINFORCE with baseline. The reward is the accuracy improvement
from applying an operation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from src.search.growth import OPS, graph_features, GrowthGraph

N_FEATURES = 11  # n_nodes, n_edges, depth, total_ch, mean_ch, density, 5 prim counts
N_OPS = len(OPS)


class PolicyNetwork(nn.Module):
    """Small MLP: graph features -> operation probabilities."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_OPS),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns logits over operations."""
        return self.net(features)

    def sample(self, features: list) -> Tuple[int, torch.Tensor]:
        """Sample an operation. Returns (op_index, log_prob)."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class ValueNetwork(nn.Module):
    """Small MLP: graph features -> predicted accuracy (baseline)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PolicyTrainer:
    """Trains the policy network with REINFORCE + baseline + replay buffer."""

    def __init__(self, lr: float = 1e-3, buffer_size: int = 100):
        self.policy = PolicyNetwork()
        self.value = ValueNetwork()
        self.policy_opt = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.value_opt = torch.optim.AdamW(self.value.parameters(), lr=lr, weight_decay=1e-4)
        self.transitions: List[dict] = []  # current episode
        self.replay_buffer: List[dict] = []  # all past episodes
        self.buffer_size = buffer_size

    def select_op(self, graph: GrowthGraph, epsilon: float = 0.1) -> Tuple[int, str, list]:
        """Select an operation using epsilon-greedy.

        With probability epsilon, pick a random op (exploration).
        Otherwise, sample from the policy distribution.
        """
        features = graph_features(graph)
        if np.random.random() < epsilon:
            op_idx = np.random.randint(0, N_OPS)
            return op_idx, OPS[op_idx], features
        op_idx, _ = self.policy.sample(features)
        return op_idx, OPS[op_idx], features

    def store(self, features: list, op_idx: int, reward: float):
        """Store a transition for later training."""
        self.transitions.append({
            "features": features,
            "op_idx": op_idx,
            "reward": reward,
        })

    def update(self, entropy_coef: float = 0.01) -> dict:
        """Update policy and value networks from stored transitions + replay."""
        if not self.transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "n_transitions": 0}

        # Add current transitions to replay buffer
        self.replay_buffer.extend(self.transitions)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]

        all_transitions = self.replay_buffer

        features = torch.tensor([t["features"] for t in all_transitions], dtype=torch.float32)
        actions = torch.tensor([t["op_idx"] for t in all_transitions], dtype=torch.long)
        rewards = torch.tensor([t["reward"] for t in all_transitions], dtype=torch.float32)

        # Value loss
        predicted_values = self.value(features).squeeze(-1)
        value_loss = F.mse_loss(predicted_values, rewards)

        # Policy loss (REINFORCE with baseline + entropy regularization)
        logits = self.policy(features)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        advantage = rewards - predicted_values.detach()
        policy_loss = -(log_probs * advantage).mean()
        # Entropy bonus encourages exploration
        entropy = dist.entropy().mean()
        policy_loss = policy_loss - entropy_coef * entropy

        # Update
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.policy_opt.step()
        self.value_opt.step()

        self.transitions.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "n_transitions": len(all_transitions),
        }
