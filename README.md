# AMR Citation Network Analysis

**Complex Networks — Final Examination Project**

Automatic analysis of scientific literature on Antimicrobial Resistance (AMR),
using PubMed data. Builds a citation network, detects communities of related
research topics, and tracks their evolution over time.

## Dataset

| File | Description |
|------|-------------|
| `data/AMR_nodes.csv` | Papers (PMID, title, journal, year, descriptors, abstract) |
| `data/AMR_edges.csv` | Citations (citing PMID → cited PMID, date) |

- **~236 k nodes** (61 k on-topic + 174 k cited-only)
- **~1 M edges** (citation links)
- **Year range:** 1965–2023
- `on_topic = True` → paper retrieved via "Drug Resistance, Microbial" MeSH descriptor

## Project Structure

```
AMR_network_analysis/
├── main.py                  # CLI entrypoint — run all steps from here
├── requirements.txt
├── .gitignore
├── README.md
├── data/                    # Place AMR_nodes.csv and AMR_edges.csv here
├── results/
│   ├── figures/             # PNG plots
│   ├── communities/         # Community assignment CSVs
│   └── temporal/            # Temporal centrality CSVs
└── src/
    ├── data_loading.py      # Load & preprocess CSVs
    ├── network_builder.py   # Build igraph / NetworkX graphs
    ├── community_detection.py  # Louvain & leading-eigenvector communities
    ├── text_analysis.py     # TF-IDF topic profiling per community
    ├── temporal_analysis.py # Time-sliced networks & centrality evolution
    └── visualization.py     # All matplotlib figures
```

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Usage

Place `AMR_nodes.csv` and `AMR_edges.csv` inside the `data/` folder, then:

```bash
# Run the full pipeline
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv

# Run individual steps
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --step build
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --step communities
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --step text
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --step temporal
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --step visualize

# Work only with on-topic papers (smaller, faster)
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --on-topic-only

# Use a specific community detection method
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --method multilevel
python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --method eigenvector
```

## Pipeline Overview

1. **Build** — Construct the directed citation graph with igraph.
2. **Detect communities** — Louvain multilevel or leading-eigenvector algorithm.
3. **Text analysis** — TF-IDF over abstracts + MeSH descriptors to label each community by topic.
4. **Temporal analysis** — Build 5-year time-slice networks; track degree, betweenness, and PageRank per community over time.
5. **Visualize** — Generate network layout (DRL), community size chart, top-terms bars, and centrality evolution plots.

## Key Results (to explore)

- Community sizes and densities
- Top papers by degree in each community
- Distinctive MeSH descriptors / abstract terms per community
- Which communities grew fastest over time (e.g., ESKAPE pathogens, COVID-era AMR)

## References

- Blondel et al. (2008) — Louvain community detection
- Newman (2006) — Leading eigenvector community detection
- Fruchterman & Reingold / DrL layout for large networks
- WHO Global Action Plan on Antimicrobial Resistance (2015)
