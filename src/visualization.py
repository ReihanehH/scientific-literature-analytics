# All the figures for the project.
#
# Figures generated:
#   network_layout.png            - network coloured by community (sampled for speed)
#   community_sizes.png           - bar chart of top-N community sizes
#   top_terms_c{id}.png           - TF-IDF terms per community
#   centrality_evolution_*.png    - avg degree / betweenness / pagerank over time
#   community_growth.png          - new publications per community per period
#   wordcloud_c{id}.png           - word clouds (needs wordcloud package)

import logging
from pathlib import Path
from typing import Dict, List, Optional

import igraph as ig
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#d37295", "#fabfd2", "#8cd17d", "#b6992d", "#499894",
]

def _colors(n: int) -> List[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def plot_network_layout(
    g: ig.Graph,
    partition: "ig.VertexClustering",
    out_dir: str = "results/figures",
    sample_size: int = 8000,
    layout_algo: str = "drl",
    seed: int = 42,
) -> None:
    """
    Draw the network coloured by community.
    DRL (DrL) layout is designed for large networks; LGL also works well.
    The full 235k-node graph would take too long to render so I sample a subset.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n = g.vcount()
    if n > sample_size:
        idx            = rng.choice(n, size=sample_size, replace=False).tolist()
        subg           = g.subgraph(idx)
        membership_sub = [partition.membership[i] for i in idx]
        logger.info(f"  Layout: sampled {sample_size:,} / {n:,} nodes")
    else:
        subg           = g
        membership_sub = partition.membership

    logger.info(f"  Computing {layout_algo.upper()} layout ...")
    layout = subg.layout(layout_algo)

    n_comm        = max(membership_sub) + 1
    colors        = _colors(n_comm)
    vertex_colors = [colors[m] for m in membership_sub]

    fig, ax = plt.subplots(figsize=(14, 14))
    coords  = layout.coords
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    for edge in subg.es:
        s, t = edge.source, edge.target
        ax.plot([xs[s], xs[t]], [ys[s], ys[t]],
                color="#cccccc", linewidth=0.15, alpha=0.4, zorder=0)

    ax.scatter(xs, ys, c=vertex_colors, s=4, alpha=0.7, linewidths=0, zorder=1)

    # Legend for the top communities
    top_ids = pd.Series(membership_sub).value_counts().head(12).index.tolist()
    patches = [
        matplotlib.patches.Patch(color=colors[c], label=f"Community {c}")
        for c in top_ids
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=7, framealpha=0.8)
    ax.set_title("AMR Citation Network — Communities", fontsize=15)
    ax.axis("off")

    fig.savefig(f"{out_dir}/network_layout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved network_layout.png -> {out_dir}/")


def plot_community_sizes(
    stats_df: pd.DataFrame,
    top_n: int = 30,
    out_dir: str = "results/figures",
) -> None:
    """Horizontal bar chart of the largest communities."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    top    = stats_df.head(top_n).copy()
    colors = _colors(len(top))

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.35)))
    bars = ax.barh(
        [f"Community {r['community_id']}" for _, r in top.iterrows()],
        top["size"],
        color=colors, edgecolor="white", height=0.7,
    )
    ax.bar_label(bars, labels=[f"{v:,}" for v in top["size"]], padding=3, fontsize=8)
    ax.set_xlabel("Number of papers")
    ax.set_title(f"Top {top_n} Communities by Size")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    sns.despine(left=True)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/community_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved community_sizes.png -> {out_dir}/")


def plot_top_terms(
    top_terms_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    top_n_terms: int = 15,
    top_n_communities: int = 10,
    out_dir: str = "results/figures",
) -> None:
    """One horizontal bar chart per community showing its top TF-IDF terms."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    top_ids = stats_df.head(top_n_communities)["community_id"].tolist()
    colors  = _colors(top_n_communities)

    for i, cid in enumerate(top_ids):
        sub = top_terms_df[top_terms_df["community_id"] == cid].head(top_n_terms)
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.35)))
        ax.barh(sub["term"][::-1], sub["tfidf_score"][::-1],
                color=colors[i], edgecolor="white", height=0.7)
        ax.set_xlabel("TF-IDF score")

        size_info = stats_df[stats_df["community_id"] == cid]["size"].values
        size_str  = f"({size_info[0]:,} papers)" if len(size_info) else ""
        ax.set_title(f"Community {cid} — Top Terms {size_str}")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        sns.despine(left=True)

        fig.tight_layout()
        fig.savefig(f"{out_dir}/top_terms_c{cid}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved top_terms_c*.png for {top_n_communities} communities -> {out_dir}/")


def plot_centrality_evolution(
    evolution_df: pd.DataFrame,
    metric: str = "avg_degree",
    top_n: int = 10,
    out_dir: str = "results/figures",
) -> None:
    """Line plot showing how a centrality metric changes over time per community."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if metric not in evolution_df.columns:
        logger.warning(f"Metric '{metric}' not found in evolution_df. Skipping.")
        return

    top_comms = (
        evolution_df.groupby("community_id")["n_nodes"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    sub    = evolution_df[evolution_df["community_id"].isin(top_comms)]
    colors = _colors(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cid in enumerate(top_comms):
        c_data = sub[sub["community_id"] == cid].sort_values("period")
        if c_data.empty:
            continue
        ax.plot(c_data["period"], c_data[metric],
                marker="o", markersize=4, linewidth=1.8,
                color=colors[i], label=f"C{cid}")

    ax.set_xlabel("Period")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Temporal Evolution — {metric.replace('_', ' ').title()}")
    ax.legend(title="Community", fontsize=8, loc="upper left")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    sns.despine()

    fig.tight_layout()
    fig.savefig(f"{out_dir}/centrality_evolution_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved centrality_evolution_{metric}.png -> {out_dir}/")


def plot_community_growth(
    growth_df: pd.DataFrame,
    top_n: int = 10,
    out_dir: str = "results/figures",
) -> None:
    """Stacked area chart of new publications per community over time."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    top_comms = (
        growth_df.groupby("community_id")["new_papers"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    sub     = growth_df[growth_df["community_id"].isin(top_comms)]
    periods = sorted(sub["period"].unique())

    matrix     = np.zeros((len(top_comms), len(periods)))
    period_idx = {p: i for i, p in enumerate(periods)}
    for _, row in sub.iterrows():
        ci = top_comms.index(row["community_id"])
        pi = period_idx[row["period"]]
        matrix[ci, pi] = row["new_papers"]

    colors = _colors(top_n)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(
        range(len(periods)),
        matrix,
        labels=[f"C{c}" for c in top_comms],
        colors=colors,
        alpha=0.85,
    )
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Period")
    ax.set_ylabel("New publications")
    ax.set_title("Community Growth Over Time")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine()

    fig.tight_layout()
    fig.savefig(f"{out_dir}/community_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved community_growth.png -> {out_dir}/")


def plot_wordclouds(
    corpus: Dict[int, str],
    stats_df: pd.DataFrame,
    top_n_communities: int = 8,
    out_dir: str = "results/figures",
) -> None:
    """Generate a word cloud per community. Needs the wordcloud package."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud not installed — skipping word clouds.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    top_ids = stats_df.head(top_n_communities)["community_id"].tolist()
    colors  = _colors(top_n_communities)

    for i, cid in enumerate(top_ids):
        if cid not in corpus or not corpus[cid].strip():
            continue

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            color_func=lambda *a, **kw: colors[i],
            max_words=100,
            collocations=False,
        ).generate(corpus[cid])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        size_info = stats_df[stats_df["community_id"] == cid]["size"].values
        size_str  = f"({size_info[0]:,} papers)" if len(size_info) else ""
        ax.set_title(f"Community {cid} — Word Cloud {size_str}", fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/wordcloud_c{cid}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved wordcloud_c*.png for {top_n_communities} communities -> {out_dir}/")
