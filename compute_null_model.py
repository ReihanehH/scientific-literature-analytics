"""
Computes null-model modularity to validate community structure.

For each of N_REWIRE rewired graphs (degree-preserving double-edge swaps),
runs Louvain and records modularity. Reports mean +/- std and Z-score vs
the empirical Q = 0.746.
"""

import pickle, time
import numpy as np
import igraph as ig

N_REWIRE  = 5      # number of null-model graphs (increase for tighter CI)
N_SWAPS   = 10     # multiplier: n_swaps = N_SWAPS * edge_count

state  = pickle.load(open("results/.cache/build_state.pkl", "rb"))
g_und  = state["g_und"]
Q_emp  = 0.7464    # empirical modularity from your run

print(f"Graph: {g_und.vcount():,} nodes, {g_und.ecount():,} edges")
print(f"Empirical Q = {Q_emp:.4f}")
print(f"Running {N_REWIRE} null-model rewirings ...\n")

q_null = []
for i in range(N_REWIRE):
    t0  = time.time()
    g_r = g_und.copy()
    g_r.rewire(n=N_SWAPS * g_r.ecount())          # degree-preserving rewiring
    part = g_r.community_multilevel(return_levels=False)
    q    = part.modularity
    q_null.append(q)
    print(f"  Run {i+1}/{N_REWIRE}: Q_null = {q:.4f}  ({time.time()-t0:.0f}s)")

q_null = np.array(q_null)
mean_q = q_null.mean()
std_q  = q_null.std()
z      = (Q_emp - mean_q) / std_q if std_q > 0 else float("inf")

print(f"\n=== Results ===")
print(f"Null-model Q: {mean_q:.4f} ± {std_q:.4f}  (n={N_REWIRE})")
print(f"Empirical Q:  {Q_emp:.4f}")
print(f"Z-score:      {z:.1f}")

