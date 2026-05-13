from typing import Optional
import pandas as pd
from exclusion_log import ExclusionLog


def get_duplicates_to_drop(
    duplicates: pd.DataFrame,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
    exclusion_step: str = "fuzzy_title_dedup",
) -> Optional[pd.DataFrame]:
    if selection_method == "auto":
        to_keep_idx = None
        for preferred_type in ["article", "Conference Paper", "preprint"]:
            candidates = duplicates[duplicates["doc_type"] == preferred_type]
            if len(candidates) > 0:
                to_keep_idx = candidates.sort_values(by="publication_date").iloc[0].name
                break
        if to_keep_idx is None:
            to_keep_idx = duplicates.iloc[0].name

        kept = duplicates.loc[to_keep_idx]
        to_drop = duplicates.drop(to_keep_idx)

        if exclusion_log is not None:
            for _, row in to_drop.iterrows():
                dropped_doc_type = str(row.get("doc_type", ""))
                dropped_doi = str(row.get("doi", ""))
                kept_source_id = str(kept.get("source_id", ""))
                kept_doc_type = str(kept.get("doc_type", ""))
                if dropped_doc_type == "preprint":
                    detail = (
                        f"Preprint dropped in favor of peer-reviewed version "
                        f"(kept source_id={kept_source_id}, type={kept_doc_type}); "
                        f"preprint DOI={dropped_doi}"
                    )
                else:
                    detail = (
                        f"Duplicate dropped; kept source_id={kept_source_id} "
                        f"(type={kept_doc_type})"
                    )
                exclusion_log.add(
                    source_id=str(row.get("source_id", "")),
                    doi=dropped_doi,
                    title=str(row.get("title", "")),
                    exclusion_step=exclusion_step,
                    exclusion_detail=detail,
                    duplicate_kept_id=kept_source_id,
                    decided_by="auto",
                )

        return to_drop

    elif selection_method == "human":
        item_reference: dict[int, pd.Series] = {}
        for num, (idx, item) in enumerate(duplicates.iterrows()):
            item_reference[num] = item

        print("\nDuplicate items found:")
        for key, item in item_reference.items():
            print(
                f"  [{key}] source_id={item.get('source_id', 'N/A')}, "
                f"type={item.get('doc_type', 'N/A')}, "
                f"date={item.get('publication_date', 'N/A')}, "
                f"doi={item.get('doi', 'N/A')}, "
                f"title={str(item.get('title', 'N/A'))[:80]!r}"
            )

        raw = input("Enter item numbers to drop (comma-separated), or press Enter to skip: ").strip()
        if not raw:
            return pd.DataFrame(columns=duplicates.columns)

        to_drop_source_ids: list[str] = []
        for key in raw.split(","):
            key = key.strip()
            if key.isdigit() and int(key) in item_reference:
                to_drop_source_ids.append(str(item_reference[int(key)].get("source_id", "")))
            else:
                print(f"  Invalid input: {key!r}. Skipping.")

        to_drop = duplicates[duplicates["source_id"].isin(to_drop_source_ids)]
        kept = duplicates[~duplicates["source_id"].isin(to_drop_source_ids)]
        kept_id_str = ", ".join(str(s) for s in kept["source_id"].tolist())

        if exclusion_log is not None:
            for _, row in to_drop.iterrows():
                exclusion_log.add(
                    source_id=str(row.get("source_id", "")),
                    doi=str(row.get("doi", "")),
                    title=str(row.get("title", "")),
                    exclusion_step=exclusion_step,
                    exclusion_detail=f"Human reviewer decision; kept source_id={kept_id_str}",
                    duplicate_kept_id=kept_id_str,
                    decided_by="human",
                )

        return to_drop

    return None


def find_doi_duplicates_to_drop(
    library_df: pd.DataFrame,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    duplicates_df = pd.DataFrame(columns=library_df.columns)

    doi_mask = (
        library_df.duplicated(subset=["doi"], keep=False)
        & library_df["doi"].notna()
        & (library_df["doi"] != "")
    )
    doi_dupes = library_df[doi_mask]

    if doi_dupes.empty:
        return duplicates_df

    for _doi, group in doi_dupes.groupby("doi"):
        to_drop = get_duplicates_to_drop(
            group,
            selection_method=selection_method,
            exclusion_log=exclusion_log,
            exclusion_step="doi_dedup",
        )
        if to_drop is not None and not to_drop.empty:
            duplicates_df = pd.concat([duplicates_df, to_drop])

    return duplicates_df


def find_exact_title_duplicates_to_drop(
    library_df: pd.DataFrame,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    title_dupes = library_df[library_df.duplicated(subset=["title"], keep=False)]

    if title_dupes.empty:
        return pd.DataFrame(columns=library_df.columns)

    duplicates_df = pd.DataFrame(columns=library_df.columns)

    for _title, group in title_dupes.groupby("title"):
        to_drop = get_duplicates_to_drop(
            group,
            selection_method=selection_method,
            exclusion_log=exclusion_log,
            exclusion_step="exact_title_dedup",
        )
        if to_drop is not None and not to_drop.empty:
            duplicates_df = pd.concat([duplicates_df, to_drop])

    return duplicates_df


def find_all_duplicates_to_drop(
    duplicates: list[list[str]],
    library_df: pd.DataFrame,
    selection_method: str = "human",
    exclusion_log: Optional[ExclusionLog] = None,
) -> pd.DataFrame:
    duplicates_df = pd.DataFrame(columns=library_df.columns)

    for duplicate_key_group in duplicates:
        group_df = library_df[library_df["source_id"].isin(duplicate_key_group)]
        to_drop = get_duplicates_to_drop(
            group_df,
            selection_method=selection_method,
            exclusion_log=exclusion_log,
            exclusion_step="fuzzy_title_dedup",
        )
        if to_drop is not None and not to_drop.empty:
            duplicates_df = pd.concat([duplicates_df, to_drop])

    return duplicates_df
