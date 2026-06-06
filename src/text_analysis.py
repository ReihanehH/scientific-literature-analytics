# TF-IDF topic profiling per community.
#
# For each community I concatenate all text (abstracts + MeSH descriptors +
# author keywords) into one big "document", then fit TF-IDF treating each
# community as a document. The terms with the highest scores are the most
# distinctive for that community compared to all others.

import logging
import re
import string
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# These are too common in the AMR corpus to be informative
EXTRA_STOPWORDS = {
    "study", "studies", "using", "used", "use", "also", "among",
    "including", "result", "results", "data", "analysis", "based",
    "method", "methods", "significant", "associated", "showed",
    "increase", "decreased", "high", "low", "compared", "found",
    "different", "may", "two", "three", "patients", "clinical",
    "infection", "infections", "antimicrobial", "resistance",
    "drug", "drugs", "bacterial", "bacteria", "strains", "strain",
}


def build_community_corpus(
    nodes_df: pd.DataFrame,
    community_col: str = "community_id",
    use_abstract: bool = True,
    use_descriptors: bool = True,
    use_keywords: bool = True,
    min_community_size: int = 10,
) -> Dict[int, str]:
    """
    Aggregate all text from papers in each community into one string per community.
    Communities smaller than min_community_size are skipped (too little text).
    """
    if community_col not in nodes_df.columns:
        raise ValueError(f"Column '{community_col}' not found. Run assign_communities first.")

    corpus: Dict[int, str] = {}

    for cid, group in nodes_df.groupby(community_col):
        if len(group) < min_community_size:
            continue

        parts = []
        if use_abstract and "AbstractText" in group.columns:
            parts.extend(group["AbstractText"].fillna("").tolist())
        if use_descriptors and "Descriptors" in group.columns:
            for desc_list in group["Descriptors"]:
                if isinstance(desc_list, list):
                    # filter out any None items that sneak into the lists
                    parts.append(" ".join(d for d in desc_list if isinstance(d, str)))
        if use_keywords and "Keywords" in group.columns:
            for kw_list in group["Keywords"]:
                if isinstance(kw_list, list):
                    parts.append(" ".join(k for k in kw_list if isinstance(k, str)))

        corpus[int(cid)] = _clean_text(" ".join(parts))

    logger.info(f"Built corpus for {len(corpus)} communities (min size={min_community_size})")
    return corpus


def fit_tfidf(
    corpus: Dict[int, str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    """
    Fit TF-IDF with each community as one document.
    Returns the fitted vectorizer and a communities x terms matrix.
    """
    community_ids = list(corpus.keys())
    documents     = [corpus[cid] for cid in community_ids]

    # Combine sklearn's English stopwords with our domain-specific ones
    all_stopwords = list(
        TfidfVectorizer(stop_words="english").get_stop_words() | EXTRA_STOPWORDS
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=all_stopwords,
        sublinear_tf=True,       # use log(1+tf) to reduce impact of very frequent terms
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )

    matrix   = vectorizer.fit_transform(documents)
    tfidf_df = pd.DataFrame(
        matrix.toarray(),
        index=community_ids,
        columns=vectorizer.get_feature_names_out(),
    )
    tfidf_df.index.name = "community_id"

    logger.info(f"TF-IDF: {tfidf_df.shape[0]} communities x {tfidf_df.shape[1]} terms")
    return vectorizer, tfidf_df


def top_terms_per_community(tfidf_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the top n TF-IDF terms for each community as a flat DataFrame."""
    rows = []
    for cid in tfidf_df.index:
        scores = tfidf_df.loc[cid].sort_values(ascending=False).head(n)
        for rank, (term, score) in enumerate(scores.items(), start=1):
            rows.append({
                "community_id": cid,
                "rank":         rank,
                "term":         term,
                "tfidf_score":  round(float(score), 5),
            })
    return pd.DataFrame(rows)


def top_descriptors_per_community(
    nodes_df: pd.DataFrame,
    community_col: str = "community_id",
    n: int = 15,
    min_community_size: int = 10,
) -> pd.DataFrame:
    """
    Count the most frequent MeSH descriptors per community (raw frequency).
    Useful as a quick, interpretable topic label that doesn't need TF-IDF.
    """
    if "Descriptors" not in nodes_df.columns:
        raise ValueError("'Descriptors' column not found.")

    rows = []
    for cid, group in nodes_df.groupby(community_col):
        if len(group) < min_community_size:
            continue
        counter: Dict[str, int] = {}
        for desc_list in group["Descriptors"]:
            if isinstance(desc_list, list):
                for d in desc_list:
                    if isinstance(d, str):
                        counter[d] = counter.get(d, 0) + 1
        top = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:n]
        for rank, (desc, cnt) in enumerate(top, start=1):
            rows.append({"community_id": cid, "descriptor": desc, "count": cnt, "rank": rank})
    return pd.DataFrame(rows)


def save_text_results(
    top_terms_df: pd.DataFrame,
    top_desc_df: Optional[pd.DataFrame] = None,
    out_dir: str = "results/communities",
) -> None:
    """Write TF-IDF terms and MeSH descriptor counts to CSV."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    top_terms_df.to_csv(f"{out_dir}/top_tfidf_terms.csv", index=False)
    logger.info(f"Saved top_tfidf_terms.csv -> {out_dir}/")
    if top_desc_df is not None:
        top_desc_df.to_csv(f"{out_dir}/top_mesh_descriptors.csv", index=False)
        logger.info(f"Saved top_mesh_descriptors.csv -> {out_dir}/")


def print_top_terms(top_terms_df: pd.DataFrame, n_communities: int = 10) -> None:
    """Quick console preview of the top terms per community."""
    for cid in sorted(top_terms_df["community_id"].unique())[:n_communities]:
        sub   = top_terms_df[top_terms_df["community_id"] == cid]
        terms = ", ".join(sub.head(10)["term"].tolist())
        print(f"  Community {cid:>4}: {terms}")


# --- helpers ---

def _clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text
