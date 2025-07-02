import networkx as nx

def build_olfactory_circuit():
    G = nx.DiGraph()

    # Add olfactory circuit neurons
    G.add_node("AL_PN", nt="ach", neuropil="AL", type="input")
    G.add_node("LH1", nt="ach", neuropil="LH", type="relay")
    G.add_node("MB_KC", nt="ach", neuropil="MB", type="integrator")
    G.add_node("PAM_DA", nt="da", neuropil="MB", type="modulator")
    G.add_node("NPF1", nt="ach", neuropil="MB", type="state_mod")

    # Add edges (with representative synaptic weights)
    G.add_edge("AL_PN", "LH1", weight=10)
    G.add_edge("LH1", "MB_KC", weight=8)
    G.add_edge("NPF1", "PAM_DA", weight=6)
    G.add_edge("PAM_DA", "MB_KC", weight=9)
    G.add_edge("MB_KC", "LH1", weight=4)  # feedback

    return G
