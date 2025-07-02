
import torch

def load_connectome_sparse_matrix(filepath):
    edge_list = []
    node_set = set()
    with open(filepath, 'r') as f:
        for line in f:
            s, t = line.strip().split()
            edge_list.append((int(s), int(t)))
            node_set.update([int(s), int(t)])

    node_list = sorted(node_set)
    node_map = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)

    rows = [node_map[s] for s, t in edge_list]
    cols = [node_map[t] for s, t in edge_list]
    values = [1.0] * len(edge_list)

    indices = torch.tensor([rows, cols])
    values = torch.tensor(values)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    return sparse_matrix, node_map
