from typing import Optional
import pandas as pd


def find_exact_title_duplicates_to_drop(library_df) -> pd.DataFrame:
    title_duplicates = library_df[library_df.duplicated(subset=["title"], keep=False)]

    if title_duplicates.empty:
        return pd.DataFrame(columns=library_df.columns)

    grouped = title_duplicates.groupby("title")
    to_drop_df = pd.DataFrame(columns=library_df.columns, dtype=str)

    for title, group in grouped:
        # Priority: article > proceedings-like > first row
        if any(group["doc_type"] == "article"):
            to_keep = group[group["doc_type"] == "article"].iloc[0]
        elif any(group["doc_type"].str.contains("Conference Paper", case=False, na=False)):
            to_keep = group[group["doc_type"].str.contains("Conference Paper", case=False, na=False)].iloc[0]
        else:
            to_keep = group.iloc[0]

        to_drop_from_group = group[group.index != to_keep.name]
        to_drop_df = pd.concat([to_drop_df, to_drop_from_group])

    return to_drop_df


def get_duplicates_to_drop(duplicates: pd.DataFrame, selection_method="auto") -> Optional[pd.DataFrame]:
    if selection_method == "auto":
        for preferred_type in ["article", "proceedings-article", "preprint"]:
            candidates = duplicates[duplicates["doc_type"] == preferred_type]
            if len(candidates) > 0:
                to_keep_idx = candidates.sort_values(by="publication_date").iloc[0].name
                return duplicates.drop(to_keep_idx)
        # fallback: keep first
        return duplicates.iloc[1:]

    elif selection_method == "human":
        item_reference = {}
        for num, (idx, item) in enumerate(duplicates.iterrows()):
            item_reference[num] = item
        print("Duplicate items found:")
        for key, item in item_reference.items():
            print(
                f"Item {key}: ID: {item['source_id']}, Title: {item['title']}, "
                f"Type: {item['doc_type']}, Publication Date: {item['publication_date']}"
            )
        to_drop_keys = input("Enter the item numbers to drop, separated by commas: ")
        to_drop_indices = []
        for key in to_drop_keys.split(","):
            key = key.strip()
            if key.isdigit() and int(key) in item_reference:
                to_drop_indices.append(item_reference[int(key)].source_id)
            else:
                print(f"Invalid input: {key}. Skipping.")
        return duplicates[duplicates["source_id"].isin(to_drop_indices)]
    return None


def find_all_duplicates_to_drop(duplicates: list[list[str]], library_df, selection_method="auto") -> pd.DataFrame:
    duplicates_df = pd.DataFrame(columns=library_df.columns, dtype=str)

    for duplicate_key_group in duplicates:
        duplicates_group_df = library_df[library_df["source_id"].isin(duplicate_key_group)]
        to_drop = get_duplicates_to_drop(duplicates_group_df, selection_method=selection_method)
        if to_drop is not None:
            duplicates_df = pd.concat([duplicates_df, to_drop])

    return duplicates_df