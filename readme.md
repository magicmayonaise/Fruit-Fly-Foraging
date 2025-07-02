# 🧠 Flybody Connectome-Driven Foraging Agent

This project extends [TuragaLab's ](https://github.com/TuragaLab/flybody)[`flybody`](https://github.com/TuragaLab/flybody) to model Drosophila foraging behavior under internal states (e.g. hunger) using connectomic constraints from the **FlyWire** Drosophila brain wiring diagram.

It demonstrates a biologically inspired control agent whose movement is:

- Modulated by internal hunger state
- Constrained by a mushroom-body-like connectome
- Trained and tested in a foraging arena with resource goals

---

## 📁 Project Structure

```
flybody-extension/
├── main.py                           # Runs simulation episode
├── models/
│   └── policy_net.py                 # PyTorch sparse-policy model
├── envs/
│   └── arena.py                      # 2D arena with food rewards
├── utils/
│   └── flywire_helpers.py           # Load FlyWire edgelist
├── data/
│   └── flywire_mb.edgelist          # Example sparse connectome graph
├── notebooks/
│   └── visualize_policy.ipynb       # Visualization notebook
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
$ git clone https://github.com/magicmayonaise/flybody-extension
$ cd flybody-extension

# Set up environment
$ pip install -r requirements.txt

# Run simulation
$ python main.py
```

---

## 🧠 Biological Background

- The model is inspired by **Kenyon cells**, **MBONs**, and **dopamine neurons (DANs)** in the fly brain.
- FlyWire’s connectome is used to structure part of the policy network.
- The agent's motion reflects hunger-dependent search—mimicking fly foraging dynamics under metabolic state changes.

---

## 📊 Visual Outputs

- Simulated trajectory in 2D space
- Neural activity patterns
- Comparison of agent vs. real Drosophila behavior (from tracked video)

---

## 📂 Example `.edgelist`

Place your `flywire_mb.edgelist` in `data/`, using:

```
0 1
1 2
2 3
3 4
```

Each line is a directed edge: neuron A → neuron B

---

## 📚 References

- [FlyWire Connectome](https://flywire.ai/)
- Bennett et al. (2021), *Learning in the Mushroom Body*
- TuragaLab `flybody` repo

---

## 🧬 Author

[Dan Landayan](https://www.danlandayan.com) – PhD in Neuroscience – modeling neural systems and biological learning algorithms.

