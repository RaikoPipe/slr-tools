#!/usr/bin/env python3
"""SLR cleaning pipeline — aligned to slr_consolidated_results.csv schema."""

import argparse
import os
from typing import Optional

import pandas as pd

from defuzzer import find_all_fuzzy_duplicated_titles
from deduplicator import find_doi_duplicates_to_drop, find_exact_title_duplicates_to_drop
from exclusion_log import ExclusionLog


VALID_DOC_TYPES = ["preprint", "article", "book-chapter", "report", "book", "dissertation"]
VALID_LANGUAGES = {"en", "de"}


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def drop_doi_duplicates(
    df: pd.DataFrame,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    to_drop = find_doi_duplicates_to_drop(df, selection_method=selection_method, exclusion_log=exclusion_log)
    return df.drop(to_drop.index)


def drop_exact_title_duplicates(
    df: pd.DataFrame,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    to_drop = find_exact_title_duplicates_to_drop(df, selection_method=selection_method, exclusion_log=exclusion_log)
    return df.drop(to_drop.index)


def drop_fuzzy_title_duplicates(
    df: pd.DataFrame,
    fuzzy_threshold: int = 80,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    to_drop = find_all_fuzzy_duplicated_titles(
        df, threshold=fuzzy_threshold, selection_method=selection_method, exclusion_log=exclusion_log
    )
    return df.drop(to_drop.index)


def drop_retracted(
    df: pd.DataFrame,
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    if "is_retracted" not in df.columns:
        return df
    mask = df["is_retracted"] == True  # noqa: E712 — intentional equality for bool column
    excluded = df[mask]
    if exclusion_log is not None:
        for _, row in excluded.iterrows():
            exclusion_log.add(
                source_id=str(row.get("source_id", "")),
                doi=str(row.get("doi", "")),
                title=str(row.get("title", "")),
                exclusion_step="retracted",
                exclusion_detail="Record flagged as retracted (is_retracted=True)",
                decided_by="auto",
            )
    return df[~mask]


def drop_before_year(
    df: pd.DataFrame,
    min_year: int = 2022,
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    mask = df["publication_year"] < min_year
    excluded = df[mask]
    if exclusion_log is not None:
        for _, row in excluded.iterrows():
            exclusion_log.add(
                source_id=str(row.get("source_id", "")),
                doi=str(row.get("doi", "")),
                title=str(row.get("title", "")),
                exclusion_step="year",
                exclusion_detail=f"publication_year {row.get('publication_year', 'N/A')} < {min_year}",
                decided_by="auto",
            )
    return df[~mask]


def drop_invalid_types(
    df: pd.DataFrame,
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    mask = ~df["doc_type"].isin(VALID_DOC_TYPES)
    excluded = df[mask]
    if exclusion_log is not None:
        for _, row in excluded.iterrows():
            exclusion_log.add(
                source_id=str(row.get("source_id", "")),
                doi=str(row.get("doi", "")),
                title=str(row.get("title", "")),
                exclusion_step="doc_type",
                exclusion_detail=f"doc_type '{row.get('doc_type', '')}' not in accepted list",
                decided_by="auto",
            )
    return df[~mask]


def drop_excluded_languages(
    df: pd.DataFrame,
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    mask = ~df["language"].isin(VALID_LANGUAGES)
    excluded = df[mask]
    if exclusion_log is not None:
        for _, row in excluded.iterrows():
            exclusion_log.add(
                source_id=str(row.get("source_id", "")),
                doi=str(row.get("doi", "")),
                title=str(row.get("title", "")),
                exclusion_step="language",
                exclusion_detail=f"language '{row.get('language', '')}' not in {sorted(VALID_LANGUAGES)}",
                decided_by="auto",
            )
    return df[~mask]


def run_pipeline(
    input_path: str,
    output_path: str,
    min_year: int = 2022,
    fuzzy_threshold: int = 80,
    selection_method: str = "human",
) -> None:
    df = load(input_path)
    n_start = len(df)
    print(f"Loaded {n_start} papers from {input_path}")

    log = ExclusionLog()

    steps = [
        ("DOI duplicates",         lambda d: drop_doi_duplicates(d, selection_method, log)),
        ("exact title duplicates", lambda d: drop_exact_title_duplicates(d, selection_method, log)),
        ("fuzzy title duplicates", lambda d: drop_fuzzy_title_duplicates(d, fuzzy_threshold, selection_method, log)),
        ("retracted",              lambda d: drop_retracted(d, log)),
        (f"year < {min_year}",     lambda d: drop_before_year(d, min_year, log)),
        ("invalid type",           lambda d: drop_invalid_types(d, log)),
        ("excluded language",      lambda d: drop_excluded_languages(d, log)),
    ]

    for label, fn in steps:
        before = len(df)
        df = fn(df)
        print(f"  Dropped {before - len(df):>4} papers — {label}")

    print(f"Final: {len(df)} papers ({n_start - len(df)} removed)")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    log_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "exclusion_log.csv")
    log.to_csv(log_path)
    print(f"Exclusion log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLR cleaning pipeline")
    parser.add_argument("input", help="Path to consolidated CSV")
    parser.add_argument("-o", "--output", default="cleaned.csv", help="Output CSV path")
    parser.add_argument("--min-year", type=int, default=2022)
    parser.add_argument("--fuzzy-threshold", type=int, default=80)
    parser.add_argument(
        "--selection-method",
        choices=["auto", "human"],
        default="human",
        help="Duplicate resolution mode: 'human' (default) prompts reviewer; 'auto' uses type-priority heuristic",
    )
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.min_year, args.fuzzy_threshold, args.selection_method)
