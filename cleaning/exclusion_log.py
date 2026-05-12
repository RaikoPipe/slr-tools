import pandas as pd

# Maps pipeline step names to coded exclusion reasons (per OSF screening instructions §1–10)
REASON_CODES: dict[str, int] = {
    "doi_dedup": 1,
    "exact_title_dedup": 1,
    "fuzzy_title_dedup": 1,
    "retracted": 2,
    "year": 3,
    "doc_type": 4,
    "language": 5,
}


class ExclusionLog:
    COLUMNS = [
        "source_id",
        "doi",
        "title",
        "exclusion_step",
        "exclusion_reason_code",
        "exclusion_detail",
        "duplicate_kept_id",
        "decided_by",
    ]

    def __init__(self) -> None:
        self._records: list[dict] = []

    def add(
        self,
        *,
        source_id: str,
        doi: str,
        title: str,
        exclusion_step: str,
        exclusion_detail: str,
        duplicate_kept_id: str = "",
        decided_by: str = "auto",
    ) -> None:
        self._records.append({
            "source_id": source_id,
            "doi": doi,
            "title": title,
            "exclusion_step": exclusion_step,
            "exclusion_reason_code": REASON_CODES.get(exclusion_step, ""),
            "exclusion_detail": exclusion_detail,
            "duplicate_kept_id": duplicate_kept_id,
            "decided_by": decided_by,
        })

    def to_csv(self, path: str) -> None:
        pd.DataFrame(self._records, columns=self.COLUMNS).to_csv(path, index=False)
