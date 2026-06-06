# Community detection using igraph, as recommended by the professor.
# Two methods:
#   - multilevel  : Louvain algorithm, fast even on large graphs
#   - eigenvector : leading eigenvector of the modularity matrix, slower but
#                   sometimes gives better results on smaller graphs
#
# Both methods need an undirected graph as input.

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import igraph as ig
import pandas as pd

logger = logging.getLogger(__name__)


def detect_communities(
    g: ig.Graph,
    method: str = "multilevel",
    weights: Optional[List[float]] = None,
) -> ig.VertexClustering:
    """
    Run community detection on an undirected igraph graph.
    Pass the output of igraph_to_undirected() here, not the directed graph.
    """
    if g.is_directed():
        raise ValueError(
            "Need an undirected graph. Call network_builder.igraph_to_undirected(g) first."
        )

    t0 = time.time()
    logger.info(
        f"Running community detection [{method}] on "
        f"{g.vcount():,} nodes, {g.ecount():,} edges ..."
    )

    if method == "multilevel":
        partition = g.community_multilevel(weights=weights, return_levels=False)
    elif method == "eigenvector":
        partition = g.community_leading_eigenvector(weights=weights)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'multilevel' or 'eigenvector'.")

    elapsed = time.time() - t0
    logger.info(
        f"  Found {len(partition)} communities | "
        f"modularity = {partition.modularity:.4f} | "
        f"time = {elapsed:.1f}s"
    )
    return partition


def community_stats(
    g: ig.Graph,
    partition: ig.VertexClustering,
    nodes_df: pd.DataFrame,
    pmid_to_vid: Dict[int, int],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute size, density, year distribution and top papers for each community.
    Returns a DataFrame sorted by community size (largest first).
    """
    logger.info(f"Computing stats for {len(partition)} communities ...")
    vid_to_pmid = {v: k for k, v in pmid_to_vid.items()}

    rows = []
    for cid, members in enumerate(partition):
        if len(members) == 0:
            continue

        subg    = partition.subgraph(cid)
        size    = len(members)
        density = subg.density() if size > 1 else 0.0

        pmids_in     = [vid_to_pmid.get(v) for v in members if v in vid_to_pmid]
        valid_pmids  = [p for p in pmids_in if p in nodes_df.index]
        on_topic_cnt = int(nodes_df.loc[valid_pmids, "on_topic"].sum())

        # Top papers by degree within the full graph
        degrees = sorted([(v, g.degree(v)) for v in members], key=lambda x: x[1], reverse=True)
        top_papers = []
        for vid, deg in degrees[:top_k]:
            pmid = vid_to_pmid.get(vid)
            if pmid and pmid in nodes_df.index:
                row = nodes_df.loc[pmid]
                top_papers.append({
                    "pmid":    pmid,
                    "degree":  deg,
                    "title":   str(row.get("Article", ""))[:120],
                    "year":    int(row["Year"]) if pd.notna(row["Year"]) else None,
                    "journal": str(row.get("JournalTitle", ""))[:60],
                })

        years = [
            int(nodes_df.loc[p, "Year"])
            for p in valid_pmids
            if pd.notna(nodes_df.loc[p, "Year"])
        ]
        median_year = int(sorted(years)[len(years) // 2]) if years else None

        rows.append({
            "community_id":   cid,
            "size":           size,
            "on_topic_count": on_topic_cnt,
            "density":        round(density, 6),
            "internal_edges": subg.ecount(),
            "median_year":    median_year,
            "top_papers":     top_papers,
        })

    stats_df = (
        pd.DataFrame(rows)
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(f"  Largest community: {stats_df['size'].iloc[0]:,} nodes")
    return stats_df


def assign_communities(
    nodes_df: pd.DataFrame,
    partition: ig.VertexClustering,
    pmid_to_vid: Dict[int, int],
    g: ig.Graph,
) -> pd.DataFrame:
    """
    Add community_id and degree columns to nodes_df.
    This is what I use to connect the igraph partition back to the pandas world.
    """
    vid_to_pmid = {v: k for k, v in pmid_to_vid.items()}

    comm_map   = {}
    degree_map = {}
    for vid, cid in enumerate(partition.membership):
        pmid = vid_to_pmid.get(vid)
        if pmid is not None:
            comm_map[pmid]   = cid
            degree_map[pmid] = g.degree(vid)

    result = nodes_df.copy()
    result["community_id"] = result.index.map(comm_map)
    result["degree"]       = result.index.map(degree_map)
    return result


def save_community_results(
    assigned_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: str = "results/communities",
) -> None:
    """Save community assignments and per-community stats to CSV."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cols = [c for c in ["Article", "Year", "JournalTitle", "on_topic", "community_id", "degree"]
            if c in assigned_df.columns]
    assigned_df[cols].to_csv(f"{out_dir}/community_assignments.csv")
    logger.info(f"Saved community_assignments.csv -> {out_dir}/")

    # Drop the nested top_papers list for the flat CSV
    stats_slim = stats_df.drop(columns=["top_papers"], errors="ignore")
    stats_slim.to_csv(f"{out_dir}/community_stats.csv", index=False)
    logger.info(f"Saved community_stats.csv -> {out_dir}/")

    # Flatten top papers into a separate file
    rows = []
    for _, row in stats_df.iterrows():
        for rank, paper in enumerate(row.get("top_papers", []), start=1):
            rows.append({"community_id": row["community_id"], "rank": rank, **paper})
    if rows:
        pd.DataFrame(rows).to_csv(f"{out_dir}/top_papers_per_community.csv", index=False)
        logger.info(f"Saved top_papers_per_community.csv -> {out_dir}/")


def print_community_summary(stats_df: pd.DataFrame, n: int = 15) -> None:
    """Print a readable summary of the top n communities."""
    print(f"\n{'='*70}")
    print(f"  Top {n} communities by size")
    print(f"{'='*70}")
    print(f"{'#':>4}  {'Size':>8}  {'on_topic':>8}  {'Density':>9}  {'Med.Year':>8}")
    print("-" * 70)
    for _, row in stats_df.head(n).iterrows():
        print(
            f"{row['community_id']:>4}  {row['size']:>8,}  "
            f"{row['on_topic_count']:>8,}  {row['density']:>9.5f}  "
            f"{row['median_year'] or '?':>8}"
        )
        for paper in row["top_papers"][:3]:
            title = (paper["title"][:65] + "…") if len(paper["title"]) > 65 else paper["title"]
            print(f"      [{paper['year'] or '?'}] deg={paper['degree']:,} | {title}")
        print()
