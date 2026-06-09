"""
Reads the community results CSVs and prints a LaTeX longtable of
community topics (top TF-IDF terms), sorted by community size.

Run from the project root:
    python extract_topics.py > community_topics.tex
Then \input{community_topics.tex} in report.tex.
"""

import pandas as pd

TERMS_CSV = "results/communities/top_tfidf_terms.csv"
STATS_CSV = "results/communities/community_stats.csv"
TOP_N_COMMUNITIES = 20   # how many communities to include in the table
TOP_N_TERMS = 6          # how many terms to show per community


def escape(s: str) -> str:
    """Escape LaTeX special characters in a string."""
    return (
        s.replace("&", r"\&")
         .replace("%", r"\%")
         .replace("_", r"\_")
         .replace("#", r"\#")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
         .replace("$", r"\$")
    )


def main():
    terms_df = pd.read_csv(TERMS_CSV)
    stats_df = pd.read_csv(STATS_CSV).sort_values("size", ascending=False).reset_index(drop=True)

    # Build a lookup: community_id -> comma-separated top terms
    top_terms = (
        terms_df[terms_df["rank"] <= TOP_N_TERMS]
        .sort_values(["community_id", "rank"])
        .groupby("community_id")["term"]
        .apply(lambda x: ", ".join(x.tolist()))
    )

    lines = []
    lines.append(r"\begin{longtable}{clrp{9cm}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Rank} & \textbf{Community} & \textbf{Size} & \textbf{Top TF-IDF Terms} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Rank} & \textbf{Community} & \textbf{Size} & \textbf{Top TF-IDF Terms} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule \multicolumn{4}{r}{\small\itshape continued on next page} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\caption{Top " + str(TOP_N_COMMUNITIES) +
                 r" communities by size with their most distinctive TF-IDF terms.}")
    lines.append(r"\label{tab:all_topics}")
    lines.append(r"\endlastfoot")

    for rank, (_, row) in enumerate(stats_df.head(TOP_N_COMMUNITIES).iterrows(), start=1):
        cid  = int(row["community_id"])
        size = int(row["size"])
        terms_str = top_terms.get(cid, "---")
        terms_escaped = escape(terms_str)
        lines.append(
            f"{rank} & C\\,{cid} & {size:,} & \\textit{{{terms_escaped}}} \\\\"
        )

    lines.append(r"\end{longtable}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
