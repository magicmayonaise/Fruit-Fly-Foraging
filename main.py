
import torch
from models.policy_net import HungerModulatedPolicy
from envs.arena import ForagingArena
from utils.flywire_helpers import load_connectome_sparse_matrix

# Load connectome subgraph (e.g. mushroom body)
adj_matrix, neuron_map = load_connectome_sparse_matrix('data/flywire_mb.edgelist')

# Initialize hunger-modulated policy with sparse connectome
policy = HungerModulatedPolicy(input_dim=6, hidden_dim=64, output_dim=4, adjacency=adj_matrix)

# Initialize environment
env = ForagingArena(goal_locations=[(3.0, 3.0), (5.0, 1.0)])

# Simulate an episode
obs = env.reset(hunger_level=0.9)
total_reward = 0
for t in range(100):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    action = policy(obs_tensor)
    obs, reward, done, _ = env.step(action.detach().numpy())
    total_reward += reward
    if done:
        break

print("Episode complete. Total reward:", total_reward)
