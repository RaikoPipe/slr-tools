#!/usr/bin/env python3
"""SLR cleaning pipeline — aligned to slr_consolidated_results.csv schema."""

import argparse
import pandas as pd
from defuzzer import find_all_fuzzy_duplicated_titles
from deduplicator import find_exact_title_duplicates_to_drop


VALID_DOC_TYPES = ["preprint", "article", "book-chapter", "report", "book", "dissertation"]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def drop_duplicates(df: pd.DataFrame, fuzzy_threshold: int = 80,
                    fuzzy_selection: str = "auto") -> pd.DataFrame:
    # Exact title duplicates
    title_dups = find_exact_title_duplicates_to_drop(df)
    df = df.drop(title_dups.index)

    # Fuzzy title duplicates
    fuzzy_dups = find_all_fuzzy_duplicated_titles(
        df, threshold=fuzzy_threshold, selection_method=fuzzy_selection
    )
    df = df.drop(fuzzy_dups.index)

    # DOI duplicates (ignore empty)
    doi_mask = df.duplicated(subset=["doi"], keep=False) & df["doi"].notna() & (df["doi"] != "")
    df = df.drop(df[doi_mask].index)

    return df


def drop_before_year(df: pd.DataFrame, min_year: int = 2022) -> pd.DataFrame:
    return df[df["publication_year"] >= min_year]


def drop_invalid_types(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["doc_type"].isin(VALID_DOC_TYPES)]


def drop_non_english(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["language"] == "en"]


def run_pipeline(input_path: str, output_path: str, min_year: int = 2022,
                 fuzzy_threshold: int = 80, fuzzy_selection: str = "auto") -> None:
    df = load(input_path)
    n_start = len(df)
    print(f"Loaded {n_start} papers from {input_path}")

    steps = [
        ("duplicates",    lambda d: drop_duplicates(d, fuzzy_threshold, fuzzy_selection)),
        ("year < {min_year}", lambda d: drop_before_year(d, min_year)),
        ("invalid type",  drop_invalid_types),
        ("non-English",   drop_non_english),
    ]

    for label, fn in steps:
        before = len(df)
        df = fn(df)
        print(f"  Dropped {before - len(df):>4} papers — {label}")

    print(f"Final: {len(df)} papers ({n_start - len(df)} removed)")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLR cleaning pipeline")
    parser.add_argument("input", help="Path to consolidated CSV")
    parser.add_argument("-o", "--output", default="cleaned.csv", help="Output CSV path")
    parser.add_argument("--min-year", type=int, default=2022)
    parser.add_argument("--fuzzy-threshold", type=int, default=80)
    parser.add_argument("--fuzzy-selection", choices=["auto", "human"], default="human")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.min_year,
                 args.fuzzy_threshold, args.fuzzy_selection)