# Temporal analysis of the AMR network.
#
# The approach follows the professor's suggestion:
#   1. Run community detection once on the full network (done in community_detection.py)
#   2. Build subgraphs for each 5-year window
#   3. Compute centrality measures (degree, betweenness, PageRank) per window
#   4. Aggregate by community to see which topics grew or declined over time

import logging
from pathlib import Path
from typing import Dict, List, Optional

import igraph as ig
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_temporal_graphs(
    time_slices: Dict[str, Dict[str, pd.DataFrame]],
    directed: bool = True,
) -> Dict[str, ig.Graph]:
    """Build one igraph graph per time slice."""
    graphs: Dict[str, ig.Graph] = {}

    for label, data in time_slices.items():
        n_df = data["nodes"]
        e_df = data["edges"]

        if len(n_df) == 0:
            continue

        pmids       = list(n_df.index)
        pmid_to_vid = {p: i for i, p in enumerate(pmids)}

        citing_vid = e_df["citing"].map(pmid_to_vid)
        cited_vid  = e_df["cited"].map(pmid_to_vid)
        valid      = citing_vid.notna() & cited_vid.notna()

        edge_list = list(zip(
            citing_vid[valid].astype(int).tolist(),
            cited_vid[valid].astype(int).tolist(),
        ))

        g = ig.Graph(n=len(pmids), edges=edge_list, directed=directed)
        g.vs["pmid"] = pmids
        g.vs["year"] = [int(y) if pd.notna(y) else -1 for y in n_df["Year"]]

        graphs[label] = g
        logger.debug(f"  [{label}] {g.vcount():,} nodes, {g.ecount():,} edges")

    logger.info(f"Built {len(graphs)} temporal graphs")
    return graphs


def compute_centrality_evolution(
    temporal_graphs: Dict[str, ig.Graph],
    assigned_df: pd.DataFrame,
    community_col: str = "community_id",
    top_n_communities: int = 20,
    betweenness_sample: Optional[int] = 500,
) -> pd.DataFrame:
    """
    For each time slice, compute average centrality per community.

    Betweenness is O(V*E) so I use sampling for large graphs.
    Set betweenness_sample=None to compute exactly (slow).
    """
    # Track only the largest communities to keep the output readable
    top_comms = (
        assigned_df[community_col]
        .value_counts()
        .head(top_n_communities)
        .index.tolist()
    )
    logger.info(f"Tracking {len(top_comms)} communities over {len(temporal_graphs)} periods")

    rows = []
    for label, g in sorted(temporal_graphs.items()):
        logger.info(f"  Processing {label} ...")

        ug = g.as_undirected(combine_edges="ignore")

        degrees = ug.degree()

        # Approximate betweenness via vertex sampling for speed
        if betweenness_sample and ug.vcount() > betweenness_sample:
            sample_vids  = list(range(0, ug.vcount(), max(1, ug.vcount() // betweenness_sample)))
            btwn         = ug.betweenness(vertices=sample_vids, directed=False, normalized=True)
            btwn_dict    = dict(zip(sample_vids, btwn))
            btwn_values  = [btwn_dict.get(v, 0.0) for v in range(ug.vcount())]
        else:
            btwn_values = ug.betweenness(directed=False, normalized=True)

        pr_values = g.pagerank() if g.is_directed() else ug.pagerank()

        pmids      = g.vs["pmid"]
        degree_map = dict(zip(pmids, degrees))
        btwn_map   = dict(zip(pmids, btwn_values))
        pr_map     = dict(zip(pmids, pr_values))

        for cid in top_comms:
            mask          = assigned_df[community_col] == cid
            cid_pmids     = set(assigned_df.index[mask])
            present_pmids = [p for p in pmids if p in cid_pmids]

            if not present_pmids:
                continue

            rows.append({
                "period":          label,
                "community_id":    cid,
                "n_nodes":         len(present_pmids),
                "avg_degree":      float(np.mean([degree_map.get(p, 0) for p in present_pmids])),
                "avg_betweenness": float(np.mean([btwn_map.get(p, 0.0) for p in present_pmids])),
                "avg_pagerank":    float(np.mean([pr_map.get(p, 0.0) for p in present_pmids])),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Centrality evolution: {len(df)} rows")
    return df


def community_growth(
    time_slices: Dict[str, Dict[str, pd.DataFrame]],
    assigned_df: pd.DataFrame,
    community_col: str = "community_id",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Count new publications per community per time window.
    Gives a simple picture of which topics attracted more papers over time.
    """
    top_comms = (
        assigned_df[community_col]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    rows = []
    cumulative: Dict[int, int] = {c: 0 for c in top_comms}

    for label in sorted(time_slices.keys()):
        period_pmids = set(time_slices[label]["nodes"].index)

        for cid in top_comms:
            mask       = (assigned_df[community_col] == cid) & assigned_df.index.isin(period_pmids)
            new_papers = int(mask.sum())
            cumulative[cid] += new_papers
            rows.append({
                "period":            label,
                "community_id":      cid,
                "new_papers":        new_papers,
                "cumulative_papers": cumulative[cid],
            })

    return pd.DataFrame(rows)


def save_temporal_results(
    evolution_df: pd.DataFrame,
    growth_df: Optional[pd.DataFrame] = None,
    out_dir: str = "results/temporal",
) -> None:
    """Write temporal analysis results to CSV."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    evolution_df.to_csv(f"{out_dir}/centrality_evolution.csv", index=False)
    logger.info(f"Saved centrality_evolution.csv -> {out_dir}/")
    if growth_df is not None:
        growth_df.to_csv(f"{out_dir}/community_growth.csv", index=False)
        logger.info(f"Saved community_growth.csv -> {out_dir}/")
