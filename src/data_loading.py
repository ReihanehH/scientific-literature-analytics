# Functions to load and clean the two CSV files we got from PubMed:
#   AMR_nodes.csv  -> one row per paper
#   AMR_edges.csv  -> one citation per row (citing -> cited)

import ast
import logging
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def load_nodes(path: str, on_topic_only: bool = False) -> pd.DataFrame:
    """Read the nodes CSV and do basic cleaning."""
    logger.info(f"Loading nodes from {path} ...")
    df = pd.read_csv(path, index_col=0, low_memory=False)

    # Descriptors and Keywords come in as string repr of Python lists, parse them
    for col in ("Descriptors", "Keywords"):
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)

    # Year as nullable int (some papers have missing years)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Make sure AbstractText is always a string, not NaN
    df["AbstractText"] = df["AbstractText"].fillna("").astype(str)

    df = df.set_index("PMID")

    # Optionally keep only the papers retrieved via the AMR descriptor
    if on_topic_only:
        n_before = len(df)
        df = df[df["on_topic"] == True]
        logger.info(f"  on_topic filter: {n_before:,} -> {len(df):,} papers")

    _log_nodes_summary(df)
    return df


def load_edges(path: str, valid_pmids: Optional[Set[int]] = None) -> pd.DataFrame:
    """Read the edges CSV. Optionally keep only edges within a given node set."""
    logger.info(f"Loading edges from {path} ...")
    df = pd.read_csv(path, low_memory=False)

    df = df.dropna(subset=["citing", "cited"])
    df["citing"] = df["citing"].astype(int)
    df["cited"]  = df["cited"].astype(int)

    if valid_pmids is not None:
        mask = df["citing"].isin(valid_pmids) & df["cited"].isin(valid_pmids)
        n_before = len(df)
        df = df[mask].reset_index(drop=True)
        logger.info(f"  Edge filter: {n_before:,} -> {len(df):,} edges")

    logger.info(f"  Loaded {len(df):,} edges")
    return df


def make_time_slices(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    period: int = 5,
    cumulative: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split the data into time windows for the temporal analysis.

    With cumulative=False each window is independent (e.g. 2000-2004).
    With cumulative=True each window includes everything up to that year,
    so you get a growing-network view.
    """
    valid_years = nodes_df["Year"].dropna()
    min_year = int(valid_years.min())
    max_year = int(valid_years.max())

    slices: Dict[str, Dict[str, pd.DataFrame]] = {}


    for end in range(min_year + period - 1, max_year + period, period):
        start = end - period + 1
        actual_end = min(end, max_year)
        label = f"{start}-{actual_end}"

        if cumulative:
            n_mask = nodes_df["Year"] <= actual_end
        else:
            n_mask = (nodes_df["Year"] >= start) & (nodes_df["Year"] <= actual_end)

        n_slice = nodes_df[n_mask]
        pmids   = set(n_slice.index)

        e_slice = edges_df[
            edges_df["citing"].isin(pmids) & edges_df["cited"].isin(pmids)
        ].reset_index(drop=True)

        slices[label] = {"nodes": n_slice, "edges": e_slice}
        logger.debug(f"  [{label}] {len(n_slice):>7,} nodes | {len(e_slice):>9,} edges")

    logger.info(
        f"Built {len(slices)} time slices "
        f"({'cumulative' if cumulative else 'windowed'}, period={period} yr)"
    )
    return slices


# helpers

def _safe_parse_list(val) -> list:
    """Parse a Python list that was stored as a string. Return [] on failure."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s in ("", "[]", "nan"):
        return []
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def _log_nodes_summary(df: pd.DataFrame) -> None:
    on_topic = int(df["on_topic"].sum()) if "on_topic" in df.columns else "?"
    yr_min   = int(df["Year"].min()) if df["Year"].notna().any() else "?"
    yr_max   = int(df["Year"].max()) if df["Year"].notna().any() else "?"
    logger.info(
        f"  Loaded {len(df):,} nodes | on_topic={on_topic:,} | Year {yr_min}-{yr_max}"
    )
