
# === FlyWire Integration ===
from data.flywire_loader import load_flywire_data, build_olfactory_connectome
from models.olfactory_circuit import OlfactoryCircuitNet

# Load FlyWire Data
neurons, synapses, connections, ids = load_flywire_data("data/flywire")
W, id_to_idx = build_olfactory_connectome(neurons, connections)

# Initialize FlyWire-based Olfactory Circuit
olfactory_net = OlfactoryCircuitNet(W)

# Replace or integrate this into your RL agent's observation pipeline.
# Example:
# obs = env.get_observation()
# olfactory_activity = olfactory_net(torch.tensor(obs['odor_input']).float().unsqueeze(0))
# action = policy_net(olfactory_activity)

# RL training script