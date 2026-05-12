from rapidfuzz import fuzz, process
from deduplicator import find_all_duplicates_to_drop
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def find_all_fuzzy_duplicated_titles(library:pd.DataFrame, threshold=90, selection_method='auto') -> pd.DataFrame:
    duplicated_titles = []
    for idx, item in library.iterrows():
        if item['source_id'] in [item_id for group in duplicated_titles for item_id in group]:
            continue
        else:
            duplicated_titles_for_item = get_fuzzy_duplicated_titles_fuzz(item, library, threshold=threshold)
            if duplicated_titles_for_item:
                duplicated_titles.append(duplicated_titles_for_item)

    duplicates_df = find_all_duplicates_to_drop(duplicated_titles, library, selection_method=selection_method)
    return duplicates_df


def get_fuzzy_duplicated_titles_fuzz(item, library, threshold=98) -> list[str]:
    title = item['title']
    duplicated_keys = []

    for idx, row in library.iterrows():
        if row['source_id'] == item['source_id']:
            continue
        lib_title = row['title']
        score = fuzz.ratio(title, lib_title)
        if score >= threshold:
            duplicated_keys.append(row['source_id'])

    if duplicated_keys:
        duplicated_keys.append(item['source_id'])

    return duplicated_keys

# Initialize model outside the function (do this once)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_fuzzy_duplicated_titles(item, library, model, threshold=0.95) -> list[str]:
    title = item['title']
    duplicated_keys = []

    title_embedding = model.encode(title, convert_to_tensor=True)

    for idx, row in library.iterrows():
        if row['source_id'] == item['source_id']:
            continue

        lib_title = row['title']
        lib_embedding = model.encode(lib_title, convert_to_tensor=True)
        similarity = util.cos_sim(title_embedding, lib_embedding).item()

        if similarity >= threshold:
            duplicated_keys.append(row['source_id'])

    if duplicated_keys:
        duplicated_keys.append(item['source_id'])

    return duplicated_keys