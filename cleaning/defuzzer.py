from rapidfuzz import fuzz
from deduplicator import find_all_duplicates_to_drop
import pandas as pd
from typing import Optional
from exclusion_log import ExclusionLog


def find_all_fuzzy_duplicated_titles(
    library: pd.DataFrame,
    threshold: int = 90,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    duplicated_titles: list[list[str]] = []
    for _idx, item in library.iterrows():
        if item["source_id"] in [item_id for group in duplicated_titles for item_id in group]:
            continue
        dupes_for_item = get_fuzzy_duplicated_titles_fuzz(item, library, threshold=threshold)
        if dupes_for_item:
            duplicated_titles.append(dupes_for_item)

    return find_all_duplicates_to_drop(
        duplicated_titles,
        library,
        selection_method=selection_method,
        exclusion_log=exclusion_log,
    )


def get_fuzzy_duplicated_titles_fuzz(item, library: pd.DataFrame, threshold: int = 98) -> list[str]:
    title = item["title"]
    duplicated_keys: list[str] = []

    for _idx, row in library.iterrows():
        if row["source_id"] == item["source_id"]:
            continue
        score = fuzz.ratio(title, row["title"])
        if score >= threshold:
            duplicated_keys.append(row["source_id"])

    if duplicated_keys:
        duplicated_keys.append(item["source_id"])

    return duplicated_keys
