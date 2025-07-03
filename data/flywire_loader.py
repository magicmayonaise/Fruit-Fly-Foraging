
import pandas as pd
import os
import torch
import numpy as np

def load_flywire_data(base_path):
    neurons = pd.read_feather(os.path.join(base_path, "neurons.feather"))
    synapses = pd.read_feather(os.path.join(base_path, "synapses.feather"))
    connections = pd.read_feather(os.path.join(base_path, "connections.feather"))
    ids = pd.read_feather(os.path.join(base_path, "ids.feather"))
    return neurons, synapses, connections, ids

def build_olfactory_connectome(neurons, connections, target_types=None):
    if target_types is None:
        target_types = ["ORN", "PN", "LHN", "MBON"]

    olfactory_neurons = neurons[neurons['type'].str.contains('|'.join(target_types), na=False)]
    olf_ids = olfactory_neurons['id'].unique()
    sub_connections = connections[
        connections['pre_id'].isin(olf_ids) & connections['post_id'].isin(olf_ids)
    ]

    id_to_idx = {nid: i for i, nid in enumerate(olf_ids)}
    n = len(olf_ids)
    W = np.zeros((n, n))

    for _, row in sub_connections.iterrows():
        pre = id_to_idx.get(row['pre_id'])
        post = id_to_idx.get(row['post_id'])
        if pre is not None and post is not None:
            W[pre, post] += row.get('weight', 1.0)

    return torch.tensor(W, dtype=torch.float32), id_to_idx
