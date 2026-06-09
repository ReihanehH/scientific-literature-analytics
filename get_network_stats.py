"""
Print key network statistics from the build cache.
Run from the project root:
    python get_network_stats.py
"""

import pickle

state = pickle.load(open("results/.cache/build_state.pkl", "rb"))
g_dir = state["g_dir"]
g_und = state["g_und"]

indeg = g_dir.indegree()
print(f"Mean in-degree:   {sum(indeg)/len(indeg):.2f}")
print(f"Max in-degree:    {max(indeg):,}")
print(f"Avg clustering:   {g_und.transitivity_avglocal_undirected():.4f}")
print(f"Transitivity:     {g_und.transitivity_undirected():.4f}")
