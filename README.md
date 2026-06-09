# AMR Citation Network Analysis

**Complex Networks — Final Project · Reihaneh Hasani**

Citation network analysis of AMR scientific literature from PubMed. Detects research communities, profiles their topics via TF-IDF, and tracks their growth and centrality over time.

**Network:** 499,734 nodes · 1,019,656 edges · 30,086 communities · modularity Q = 0.746

## Setup

```bash
pip install -r requirements.txt
```

Place `AMR_nodes.csv` and `AMR_edges.csv` in `data/`, then:

```bash
# Full pipeline
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv

# Individual steps: build | communities | text | temporal | visualize
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --step communities

# Options
--method eigenvector   # alternative community detection (default: multilevel/Louvain)
--on-topic-only        # restrict to ~61k on-topic papers (faster)
--period 5             # years per temporal window (default: 5)
```

Results are saved to `results/figures/`, `results/communities/`, and `results/temporal/`.
