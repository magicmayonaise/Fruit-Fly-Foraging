# flybody_extension/train_rl.py

import torch
import torch.nn as nn
import torch.optim as optim
from models.policy_net import HungerModulatedPolicy
from envs.arena import ForagingArena
from utils.flywire_helpers import load_connectome_sparse_matrix

# Hyperparameters
EPOCHS = 200
EPISODE_LEN = 100
LEARNING_RATE = 1e-3

# Load connectome and environment
adj, _ = load_connectome_sparse_matrix("data/flywire_mb.edgelist")
policy = HungerModulatedPolicy(input_dim=7, hidden_dim=64, output_dim=4, adjacency=adj)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# Reward shaping weights
HUNGER_REWARD_WEIGHT = 1.0
THIRST_REWARD_WEIGHT = 1.0

for epoch in range(EPOCHS):
    env = ForagingArena(goal_locations=[(3.0, 3.0), (5.0, 1.0)], water_source=(8.0, 8.0))
    obs = env.reset(hunger_level=0.9, thirst_level=0.9)
    total_reward = 0
    log_probs = []
    rewards = []

    for _ in range(EPISODE_LEN):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_logits = policy(obs_tensor)
        dist = torch.distributions.Normal(action_logits, 0.5)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        obs, reward, done, _ = env.step(action.detach().numpy())

        # Reward shaping
        shaped_reward = HUNGER_REWARD_WEIGHT * reward * obs[2] + THIRST_REWARD_WEIGHT * reward * obs[3]

        log_probs.append(log_prob)
        rewards.append(shaped_reward)
        total_reward += reward
        if done:
            break

    # Policy gradient update
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + 0.99 * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    loss = -torch.stack(log_probs) @ returns
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Total reward: {total_reward:.2f}")
