from typing import Optional

import pandas as pd

def find_exact_title_duplicates_to_drop(library_df) -> pd.DataFrame:
    # Find rows that have duplicate titles
    title_duplicates = library_df[library_df.duplicated(subset=["title"], keep=False)]

    if title_duplicates.empty:
        # No duplicates found, return empty dataframe
        return pd.DataFrame(columns=library_df.columns)

    # handling logic for title duplicates
    grouped = title_duplicates.groupby('title')
    to_drop_df = pd.DataFrame(columns=library_df.columns, dtype=str)

    for title, group in grouped:
        # prioritize journal articles over preprint and conference papers
        if any(group['type'] == 'journal-article'):
            to_keep = group[group['type_crossref'] == 'journal-article'].iloc[0]
        elif any(group['type'] == 'proceedings-article'):
            to_keep = group[group['type_crossref'] == 'proceedings-article'].iloc[0]
        else:
            to_keep = group.iloc[0]

        # Get all rows in this group EXCEPT the one we want to keep
        to_drop_from_group = group[group.index != to_keep.name]
        to_drop_df = pd.concat([to_drop_df, to_drop_from_group])

    return to_drop_df

def get_duplicates_to_drop(duplicates: pd.DataFrame, selection_method = "auto") -> Optional[pd.DataFrame]:
    if selection_method == 'auto':
        # get rows that are journal articles
        journal_articles = duplicates[duplicates['type'] == 'article']
        preprints = duplicates[duplicates['type'] == 'preprint']
        conference_papers = duplicates[duplicates['type'] == 'proceedings-article']
        to_keep_idx = None
        for items in [journal_articles, conference_papers, preprints]:
            if len(items) > 0:
                # choose the one with the most recent publication date
                to_keep_idx = items.sort_values(by='publication_date').iloc[0].name
                break
        # return the rest as to drop
        return duplicates.drop(to_keep_idx)
    elif selection_method == 'human':
        # get selection from user as items to drop
        # create item reference dict and enumerate items
        item_reference = {}
        num = 0
        for idx, item in duplicates.iterrows():
            item_reference[num] = item
            num += 1
        print("Duplicate items found:")
        for key, item in item_reference.items():
            print(f"Item {key}: ID: {item['id']}, Title: {item['title']}, Type: {item['type']}, Publication Date: {item['publication_date']}")
        to_drop_keys = input("Enter the item numbers to drop, separated by commas: ")
        to_drop_indices = []
        for key in to_drop_keys.split(','):
            key = key.strip()
            if key.isdigit() and int(key) in item_reference:
                to_drop_indices.append(item_reference[int(key)].id)
            else:
                print(f"Invalid input: {key}. Skipping.")
        to_drop = duplicates[duplicates['id'].isin(to_drop_indices)]
        return to_drop





def find_all_duplicates_to_drop(duplicates: list[list[str]], library_df, selection_method='auto') -> pd.DataFrame:
    # construct dataframe to hold all duplicates to drop
    duplicates_df = pd.DataFrame(columns=library_df.columns, dtype=str)

    # group duplicates by list of keys
    for duplicate_key_group in duplicates:
        # get rows corresponding to duplicate keys
        duplicates_group_df = library_df[library_df['id'].isin(duplicate_key_group)]
        # apply logic to choose which to drop
        to_drop = get_duplicates_to_drop(duplicates_group_df, selection_method=selection_method)
        if to_drop is not None:
            duplicates_df = pd.concat([duplicates_df, to_drop])

    return duplicates_df