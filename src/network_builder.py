# Build the citation graph using igraph (recommended by the professor for community detection)
# and optionally NetworkX for algorithms not available in igraph.
#
# I build the igraph graph directly from the edge list instead of going through
# NetworkX -> GraphML -> igraph, which is what the professor suggested but is slow
# for a graph this size. Both routes are provided here.

import logging
from pathlib import Path
from typing import Dict, Tuple

import igraph as ig
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_igraph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    directed: bool = True,
) -> Tuple[ig.Graph, Dict[int, int]]:
    """
    Build an igraph graph directly from the DataFrames.
    Returns the graph and a PMID->vertex_id mapping (needed later to
    attach community labels back to the nodes DataFrame).
    """
    pmids = list(nodes_df.index)
    pmid_to_vid: Dict[int, int] = {pmid: i for i, pmid in enumerate(pmids)}
    n_nodes = len(pmids)

    logger.info(f"Building igraph: {n_nodes:,} nodes ...")

    # Map PMIDs to vertex indices using pandas (much faster than a Python loop)
    citing_vid = edges_df["citing"].map(pmid_to_vid)
    cited_vid  = edges_df["cited"].map(pmid_to_vid)
    valid      = citing_vid.notna() & cited_vid.notna()

    n_dropped = (~valid).sum()
    if n_dropped:
        logger.debug(f"  Dropped {n_dropped:,} edges pointing to unknown PMIDs")

    edge_list = list(zip(
        citing_vid[valid].astype(int).tolist(),
        cited_vid[valid].astype(int).tolist(),
    ))

    logger.info(f"  Adding {len(edge_list):,} edges ...")
    g = ig.Graph(n=n_nodes, edges=edge_list, directed=directed)

    # Attach the attributes we'll need later
    g.vs["pmid"]     = pmids
    g.vs["title"]    = nodes_df["Article"].fillna("").tolist()
    g.vs["year"]     = [int(y) if pd.notna(y) else -1 for y in nodes_df["Year"]]
    g.vs["on_topic"] = nodes_df["on_topic"].tolist()
    g.vs["journal"]  = nodes_df["JournalTitle"].fillna("").tolist()

    logger.info(
        f"  igraph built: {g.vcount():,} nodes, {g.ecount():,} edges "
        f"({'directed' if directed else 'undirected'})"
    )
    return g, pmid_to_vid


def igraph_to_undirected(g: ig.Graph, combine_edges: str = "ignore") -> ig.Graph:
    """
    Return an undirected copy of the graph.
    community_multilevel and community_leading_eigenvector both need undirected input.
    """
    ug = g.as_undirected(combine_edges=combine_edges)
    logger.info(f"Converted to undirected: {ug.vcount():,} nodes, {ug.ecount():,} edges")
    return ug


# --- NetworkX / GraphML route (alternative, as the professor suggested) ---

def build_networkx(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build a NetworkX directed graph from the same DataFrames."""
    logger.info(f"Building NetworkX DiGraph ({len(nodes_df):,} nodes) ...")

    G = nx.DiGraph()
    for pmid, row in nodes_df.iterrows():
        G.add_node(
            pmid,
            title=str(row.get("Article", "")),
            year=int(row["Year"]) if pd.notna(row["Year"]) else -1,
            on_topic=bool(row.get("on_topic", False)),
            journal=str(row.get("JournalTitle", "")),
        )

    pmid_set = set(nodes_df.index)
    edges = [
        (int(r["citing"]), int(r["cited"]))
        for _, r in edges_df.iterrows()
        if int(r["citing"]) in pmid_set and int(r["cited"]) in pmid_set
    ]
    G.add_edges_from(edges)

    logger.info(f"  NetworkX: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def export_graphml(G: nx.DiGraph, path: str) -> None:
    """Save to GraphML so igraph can load it with ig.Graph.Read_GraphML(path)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, path)
    logger.info(f"Saved GraphML to {path}")


def load_igraph_from_graphml(path: str) -> ig.Graph:
    """Load a graph that was previously saved as GraphML."""
    g = ig.Graph.Read_GraphML(path)
    logger.info(f"Loaded GraphML: {g.vcount():,} nodes, {g.ecount():,} edges")
    return g


# --- utilities ---

def largest_connected_component(g: ig.Graph) -> ig.Graph:
    """Return the subgraph of the largest weakly connected component."""
    comps   = g.clusters(mode="weak")
    lcc_idx = comps.sizes().index(max(comps.sizes()))
    lcc     = comps.subgraph(lcc_idx)
    logger.info(
        f"LCC: {lcc.vcount():,} / {g.vcount():,} nodes "
        f"({100 * lcc.vcount() / g.vcount():.1f}%)"
    )
    return lcc


def graph_summary(g: ig.Graph) -> dict:
    """Quick stats about the graph (nodes, edges, density, avg degree)."""
    density = (
        g.ecount() / (g.vcount() * (g.vcount() - 1))
        if g.vcount() > 1 else 0
    )
    return {
        "nodes":           g.vcount(),
        "edges":           g.ecount(),
        "directed":        g.is_directed(),
        "density":         density,
        "avg_degree":      sum(g.degree()) / g.vcount() if g.vcount() else 0,
        "components_weak": len(g.clusters(mode="weak")) if g.is_directed() else len(g.clusters()),
    }
