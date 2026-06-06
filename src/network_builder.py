# Build the citation graph using igraph (recommended by the professor for community detection)
# I build the igraph graph directly from the edge list instead of going through
# NetworkX -> GraphML -> igraph, which is what the professor suggested but is slow
# for a graph this size.

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

    All PMIDs that appear in the edge list but are not in nodes_df (i.e. papers
    cited by our papers but not collected as nodes) are added as anonymous nodes
    so that no edges are dropped. This is how we keep all 1,019,656 edges.

    Returns the graph and a PMID->vertex_id mapping (needed later to
    attach community labels back to the nodes DataFrame).
    """
    pmids = list(nodes_df.index)
    pmid_to_vid: Dict[int, int] = {pmid: i for i, pmid in enumerate(pmids)}
    n_known = len(pmids)

    # Find PMIDs referenced in edges but missing from nodes_df and add them
    # as anonymous nodes (no metadata). This preserves all edges.
    known_set = set(pmids)
    extra_pmids = (
        set(edges_df["citing"].tolist()) | set(edges_df["cited"].tolist())
    ) - known_set
    for pmid in extra_pmids:
        pmid_to_vid[pmid] = len(pmids)
        pmids.append(pmid)

    n_nodes = len(pmids)
    if extra_pmids:
        logger.info(f"  Added {len(extra_pmids):,} anonymous nodes for unlisted cited papers")

    logger.info(f"Building igraph: {n_nodes:,} nodes ({n_known:,} with metadata) ...")

    # Map all edges — no edges dropped now
    citing_vid = edges_df["citing"].map(pmid_to_vid).astype(int)
    cited_vid  = edges_df["cited"].map(pmid_to_vid).astype(int)
    edge_list  = list(zip(citing_vid.tolist(), cited_vid.tolist()))

    logger.info(f"  Adding {len(edge_list):,} edges ...")
    g = ig.Graph(n=n_nodes, edges=edge_list, directed=directed)

    # Attach metadata for the known nodes; anonymous nodes get empty/default values
    g.vs["pmid"]     = pmids
    g.vs["title"]    = (
        nodes_df["Article"].fillna("").tolist()
        + [""] * len(extra_pmids)
    )
    g.vs["year"]     = (
        [int(y) if pd.notna(y) else -1 for y in nodes_df["Year"]]
        + [-1] * len(extra_pmids)
    )
    g.vs["on_topic"] = (
        nodes_df["on_topic"].tolist()
        + [False] * len(extra_pmids)
    )
    g.vs["journal"]  = (
        nodes_df["JournalTitle"].fillna("").tolist()
        + [""] * len(extra_pmids)
    )

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


# NetworkX / GraphML route

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


# utilities

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
