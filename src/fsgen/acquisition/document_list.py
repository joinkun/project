"""EDINET 書類一覧の取得・絞り込み・CSV化、および書類本体(ZIP)の一括取得。

パースはしない。ZIP はそのまま `data/raw/documents/` に保存するだけ
(``fetch_documents``、内部実装は ``_document_fetch.py``)。

書類一覧APIは日付単位でしか引けないため、`date_from`〜`date_to` の全日を
1日ずつ叩く。同じ docID が複数日にまたがって現れることがある
(訂正が登録されると、訂正登録日のファイル日付にも同じ docID の行が
再掲される)。この場合は後から見つかった行(=より新しい情報)を採用する。
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from tqdm import tqdm

from fsgen.acquisition._document_fetch import FETCH_LOG_COLUMNS as FETCH_LOG_COLUMNS
from fsgen.acquisition._document_fetch import FetchLogEntry as FetchLogEntry
from fsgen.acquisition._document_fetch import fetch_documents as fetch_documents
from fsgen.acquisition._document_types import DOCUMENT_LIST_COLUMNS as DOCUMENT_LIST_COLUMNS
from fsgen.acquisition._document_types import DocumentRecord as DocumentRecord
from fsgen.acquisition.edinet_client import EdinetApiError, EdinetClient

log = logging.getLogger(__name__)

__all__ = [
    "DOCUMENT_LIST_COLUMNS",
    "FETCH_LOG_COLUMNS",
    "DocumentListSummary",
    "DocumentRecord",
    "FetchLogEntry",
    "fetch_document_list",
    "fetch_documents",
    "iter_dates",
    "write_document_list_csv",
]


@dataclass(frozen=True)
class DocumentListSummary:
    records: tuple[DocumentRecord, ...]
    days_queried: int
    raw_result_count: int
    duplicate_edinet_codes: dict[str, int]
    failed_dates: tuple[date, ...]

    @property
    def unique_document_count(self) -> int:
        return len(self.records)

    @property
    def unique_edinet_code_count(self) -> int:
        return len({r.edinetCode for r in self.records if r.edinetCode})


def iter_dates(date_from: date, date_to: date) -> Iterator[date]:
    if date_from > date_to:
        raise ValueError(f"date_from ({date_from}) must not be after date_to ({date_to})")
    current = date_from
    while current <= date_to:
        yield current
        current += timedelta(days=1)


def fetch_document_list(
    client: EdinetClient,
    date_from: date,
    date_to: date,
    doc_type_codes: tuple[str, ...],
    ordinance_codes: tuple[str, ...],
    require_xbrl: bool,
    *,
    show_progress: bool = True,
) -> DocumentListSummary:
    dates = list(iter_dates(date_from, date_to))
    records: dict[str, DocumentRecord] = {}
    raw_result_count = 0
    failed_dates: list[date] = []

    for target_date in tqdm(dates, desc="documents.json", disable=not show_progress):
        try:
            response = client.list_documents(target_date)
        except EdinetApiError as exc:
            log.error("failed to list documents for %s: %s", target_date, exc)
            failed_dates.append(target_date)
            continue

        raw_result_count += len(response.results)
        for row in response.results:
            if row.get("docTypeCode") not in doc_type_codes:
                continue
            if row.get("ordinanceCode") not in ordinance_codes:
                continue
            if require_xbrl and row.get("xbrlFlag") != "1":
                continue
            if row.get("withdrawalStatus") != "0":
                continue
            record = DocumentRecord.from_api_row(row)
            records[record.docID] = record

    edinet_counts: dict[str, int] = {}
    for record in records.values():
        if record.edinetCode:
            edinet_counts[record.edinetCode] = edinet_counts.get(record.edinetCode, 0) + 1
    duplicates = {code: count for code, count in edinet_counts.items() if count > 1}

    return DocumentListSummary(
        records=tuple(records.values()),
        days_queried=len(dates),
        raw_result_count=raw_result_count,
        duplicate_edinet_codes=duplicates,
        failed_dates=tuple(failed_dates),
    )


def write_document_list_csv(summary: DocumentListSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DOCUMENT_LIST_COLUMNS)
        writer.writeheader()
        for record in summary.records:
            writer.writerow(asdict(record))
