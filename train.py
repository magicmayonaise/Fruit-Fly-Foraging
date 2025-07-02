import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from graph_builder import build_olfactory_circuit

class SimpleNeuronModel(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()
        self.fc = nn.Linear(n_neurons, n_neurons)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def main():
    G = build_olfactory_circuit()
    neurons = list(G.nodes)
    neuron_idx = {n: i for i, n in enumerate(neurons)}

    # Build adjacency-based weight matrix
    adj = torch.zeros(len(neurons), len(neurons))
    for src, tgt, data in G.edges(data=True):
        i, j = neuron_idx[src], neuron_idx[tgt]
        adj[j, i] = data['weight'] / 10.0

    model = SimpleNeuronModel(len(neurons))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Toy input/output
    input_state = torch.zeros(len(neurons))
    input_state[neuron_idx["AL_PN"]] = 1.0  # odor stimulus

    target_output = torch.zeros(len(neurons))
    target_output[neuron_idx["MB_KC"]] = 1.0  # target: MB_KC activation

    # Train
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(input_state)
        loss = criterion(output, target_output)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("Final Output:", output)

if __name__ == "__main__":
    main()
