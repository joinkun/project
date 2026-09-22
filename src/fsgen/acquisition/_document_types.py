"""書類一覧の1行を表すデータ型。循環 import を避けるための内部モジュール。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

DOCUMENT_LIST_COLUMNS = (
    "docID",
    "edinetCode",
    "secCode",
    "filerName",
    "docTypeCode",
    "ordinanceCode",
    "periodStart",
    "periodEnd",
    "submitDateTime",
    "xbrlFlag",
    "withdrawalStatus",
    "docInfoEditStatus",
)


def _field(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


@dataclass(frozen=True)
class DocumentRecord:
    docID: str
    edinetCode: str | None
    secCode: str | None
    filerName: str | None
    docTypeCode: str | None
    ordinanceCode: str | None
    periodStart: str | None
    periodEnd: str | None
    submitDateTime: str | None
    xbrlFlag: str | None
    withdrawalStatus: str | None
    docInfoEditStatus: str | None

    @classmethod
    def from_api_row(cls, row: Mapping[str, object]) -> DocumentRecord:
        doc_id = _field(row, "docID")
        if doc_id is None:
            raise ValueError(f"document row is missing docID: {row!r}")
        return cls(
            docID=doc_id,
            edinetCode=_field(row, "edinetCode"),
            secCode=_field(row, "secCode"),
            filerName=_field(row, "filerName"),
            docTypeCode=_field(row, "docTypeCode"),
            ordinanceCode=_field(row, "ordinanceCode"),
            periodStart=_field(row, "periodStart"),
            periodEnd=_field(row, "periodEnd"),
            submitDateTime=_field(row, "submitDateTime"),
            xbrlFlag=_field(row, "xbrlFlag"),
            withdrawalStatus=_field(row, "withdrawalStatus"),
            docInfoEditStatus=_field(row, "docInfoEditStatus"),
        )
