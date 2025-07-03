# Olfactory Circuit Model (Drosophila)

Foraging simulation constrained by connectome neurobiology of an olfactory circuit using PyTorch and NetworkX.

## Components
- AL_PN → LH → MB_KC pathway (cholinergic)
- NPF and PAM neurons modulate MB activity (dopaminergic input)
- Reciprocal feedback from MB → LH

## Files
- `graph_builder.py`: Builds the NetworkX graph with FlyWire-style neuron metadata
- `train.py`: Trains a toy model to activate MB_KC from AL input
