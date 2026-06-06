# AMR Citation Network Analysis
# Complex Networks — Final Project
#
# Run the full pipeline or individual steps:
#
#   python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv
#   python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --step communities
#   python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --on-topic-only
#   python main.py --nodes data/AMR_nodes.csv --edges data/AMR_edges.csv --method eigenvector
#
# Steps: build -> communities -> text -> temporal -> visualize
# Each step caches its results so you can re-run individual steps without
# repeating the expensive ones (e.g. community detection takes ~2 min).

import argparse
import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

sys.path.insert(0, str(Path(__file__).parent))
from src import (
    data_loading,
    network_builder,
    community_detection,
    text_analysis,
    temporal_analysis,
    visualization,
)

CACHE_DIR = Path("results/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ----- pipeline steps -----

def step_build(args) -> dict:
    logger.info("=== STEP: build ===")

    nodes_df = data_loading.load_nodes(args.nodes, on_topic_only=args.on_topic_only)
    edges_df = data_loading.load_edges(args.edges, valid_pmids=set(nodes_df.index))
    logger.info(f"Nodes: {len(nodes_df):,} | Edges: {len(edges_df):,}")

    g_dir, pmid_to_vid = network_builder.build_igraph(nodes_df, edges_df, directed=True)
    g_und = network_builder.igraph_to_undirected(g_dir)

    logger.info(f"Graph summary: {network_builder.graph_summary(g_dir)}")

    state = {
        "nodes_df":    nodes_df,
        "edges_df":    edges_df,
        "g_dir":       g_dir,
        "g_und":       g_und,
        "pmid_to_vid": pmid_to_vid,
    }
    _save_cache(state, "build_state.pkl")
    return state


def step_communities(args, state: dict = None) -> dict:
    logger.info("=== STEP: communities ===")

    if state is None:
        state = _load_cache("build_state.pkl")

    nodes_df    = state["nodes_df"]
    g_und       = state["g_und"]
    pmid_to_vid = state["pmid_to_vid"]

    partition   = community_detection.detect_communities(g_und, method=args.method)
    assigned_df = community_detection.assign_communities(nodes_df, partition, pmid_to_vid, g_und)
    stats_df    = community_detection.community_stats(g_und, partition, nodes_df, pmid_to_vid, top_k=10)

    community_detection.print_community_summary(stats_df, n=20)
    community_detection.save_community_results(assigned_df, stats_df, out_dir="results/communities")

    state.update({
        "partition":   partition,
        "assigned_df": assigned_df,
        "stats_df":    stats_df,
    })
    _save_cache(state, "communities_state.pkl")
    return state


def step_text(args, state: dict = None) -> dict:
    logger.info("=== STEP: text ===")

    if state is None:
        state = _load_cache("communities_state.pkl")

    nodes_df    = state["nodes_df"]
    assigned_df = state["assigned_df"]
    stats_df    = state["stats_df"]

    # Merge community labels into the nodes DataFrame for groupby
    nodes_c = nodes_df.copy()
    nodes_c["community_id"] = assigned_df["community_id"]
    nodes_c["degree"]       = assigned_df["degree"]

    corpus       = text_analysis.build_community_corpus(nodes_c, min_community_size=20)
    _, tfidf_df  = text_analysis.fit_tfidf(corpus, max_features=8000, ngram_range=(1, 2))
    top_terms_df = text_analysis.top_terms_per_community(tfidf_df, n=25)
    top_desc_df  = text_analysis.top_descriptors_per_community(nodes_c, n=20, min_community_size=20)

    logger.info("Top terms for first 8 communities:")
    text_analysis.print_top_terms(top_terms_df, n_communities=8)

    text_analysis.save_text_results(top_terms_df, top_desc_df, out_dir="results/communities")

    state.update({
        "corpus":          corpus,
        "tfidf_df":        tfidf_df,
        "top_terms_df":    top_terms_df,
        "top_desc_df":     top_desc_df,
        "nodes_with_comm": nodes_c,
    })
    _save_cache(state, "text_state.pkl")
    return state


def step_temporal(args, state: dict = None) -> dict:
    logger.info("=== STEP: temporal ===")

    if state is None:
        state = _load_cache("text_state.pkl")

    nodes_df    = state["nodes_df"]
    edges_df    = state["edges_df"]
    assigned_df = state["assigned_df"]

    slices      = data_loading.make_time_slices(nodes_df, edges_df, period=5, cumulative=False)
    temp_graphs = temporal_analysis.build_temporal_graphs(slices, directed=True)
    growth_df   = temporal_analysis.community_growth(slices, assigned_df, top_n=15)

    logger.info("Computing centrality evolution (may take a few minutes) ...")
    evolution_df = temporal_analysis.compute_centrality_evolution(
        temp_graphs,
        assigned_df,
        top_n_communities=15,
        betweenness_sample=300,
    )

    temporal_analysis.save_temporal_results(evolution_df, growth_df, out_dir="results/temporal")

    state.update({
        "slices":       slices,
        "temp_graphs":  temp_graphs,
        "growth_df":    growth_df,
        "evolution_df": evolution_df,
    })
    _save_cache(state, "temporal_state.pkl")
    return state


def step_visualize(args, state: dict = None) -> None:
    logger.info("=== STEP: visualize ===")

    if state is None:
        state = _load_cache("temporal_state.pkl")

    out_dir = "results/figures"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    partition    = state.get("partition")
    g_und        = state.get("g_und")
    stats_df     = state["stats_df"]
    top_terms_df = state["top_terms_df"]
    corpus       = state.get("corpus", {})
    evolution_df = state.get("evolution_df")
    growth_df    = state.get("growth_df")

    if partition is not None and g_und is not None:
        logger.info("Plotting network layout (DRL, sampled) ...")
        visualization.plot_network_layout(
            g_und, partition,
            out_dir=out_dir, sample_size=6000, layout_algo="drl",
        )

    visualization.plot_community_sizes(stats_df, top_n=30, out_dir=out_dir)
    visualization.plot_top_terms(top_terms_df, stats_df,
                                 top_n_terms=15, top_n_communities=12, out_dir=out_dir)

    if evolution_df is not None and not evolution_df.empty:
        for metric in ("avg_degree", "avg_betweenness", "avg_pagerank"):
            visualization.plot_centrality_evolution(evolution_df, metric=metric,
                                                    top_n=10, out_dir=out_dir)

    if growth_df is not None and not growth_df.empty:
        visualization.plot_community_growth(growth_df, top_n=10, out_dir=out_dir)

    visualization.plot_wordclouds(corpus, stats_df, top_n_communities=8, out_dir=out_dir)

    logger.info(f"All figures saved to {out_dir}/")


# ----- orchestration -----

def run_pipeline(args) -> None:
    steps = {
        "build":       step_build,
        "communities": step_communities,
        "text":        step_text,
        "temporal":    step_temporal,
        "visualize":   step_visualize,
    }
    order = ["build", "communities", "text", "temporal", "visualize"]

    if args.step == "all":
        state = None
        for name in order:
            result = steps[name](args) if name == "build" else steps[name](args, state)
            if isinstance(result, dict):
                state = result
    else:
        fn = steps.get(args.step)
        if fn is None:
            logger.error(f"Unknown step '{args.step}'. Choose from: {list(steps)}")
            sys.exit(1)
        fn(args)

    logger.info("Done. Results saved to results/")


# ----- CLI -----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AMR Citation Network Analysis — Complex Networks Final Project"
    )
    p.add_argument("--nodes",  required=True, help="Path to AMR_nodes.csv")
    p.add_argument("--edges",  required=True, help="Path to AMR_edges.csv")
    p.add_argument(
        "--step",
        choices=["all", "build", "communities", "text", "temporal", "visualize"],
        default="all",
        help="Which step to run (default: all)",
    )
    p.add_argument(
        "--method",
        choices=["multilevel", "eigenvector"],
        default="multilevel",
        help="Community detection algorithm (default: multilevel / Louvain)",
    )
    p.add_argument(
        "--on-topic-only",
        action="store_true",
        help="Use only the ~61k on-topic papers (faster)",
    )
    p.add_argument(
        "--period", type=int, default=5,
        help="Years per time window for temporal analysis (default: 5)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


# ----- cache helpers -----

def _save_cache(obj: dict, filename: str) -> None:
    with open(CACHE_DIR / filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(filename: str) -> dict:
    path = CACHE_DIR / filename
    if not path.exists():
        logger.error(
            f"Cache not found: {path}. "
            "Run the prerequisite steps first (or 'python main.py --step all')."
        )
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run_pipeline(args)
