"""書類本体(ZIP)の一括取得。冪等性・ディスク容量チェック・ZIP検証を担う。

``document_list.py`` から内部的に使われる。公開 API はそちらの
re-export 経由で ``fsgen.acquisition.document_list`` として使うこと。
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import shutil
import sys
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from fsgen.acquisition._document_types import DocumentRecord
from fsgen.acquisition.edinet_client import EdinetApiError, EdinetClient

log = logging.getLogger(__name__)

FETCH_LOG_COLUMNS = ("docID", "status", "http_status", "bytes", "sha256", "fetched_at", "error")

# 書類取得APIでZIPをまるごと保存するので自動的に満たされるが、将来ZIPから
# 必要ファイルだけ抽出する最適化を入れたくなっても AuditDoc は残すこと。
# 監査法人名・監査人が後の分析に必要になる。
_REQUIRED_AUDIT_DOC_PREFIX = "XBRL/AuditDoc/"


@dataclass(frozen=True)
class FetchLogEntry:
    docID: str
    status: str  # "ok" | "skipped" | "failed"
    http_status: int | None
    bytes: int | None
    sha256: str | None
    fetched_at: str
    error: str | None


def _read_fetch_log(path: Path) -> dict[str, FetchLogEntry]:
    if not path.exists():
        return {}
    entries: dict[str, FetchLogEntry] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entries[row["docID"]] = FetchLogEntry(
                docID=row["docID"],
                status=row["status"],
                http_status=int(row["http_status"]) if row.get("http_status") else None,
                bytes=int(row["bytes"]) if row.get("bytes") else None,
                sha256=row.get("sha256") or None,
                fetched_at=row["fetched_at"],
                error=row.get("error") or None,
            )
    return entries


def _write_fetch_log(path: Path, entries: dict[str, FetchLogEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FETCH_LOG_COLUMNS)
        writer.writeheader()
        for entry in entries.values():
            writer.writerow(asdict(entry))
    tmp_path.replace(path)


def _validate_zip(content: bytes) -> tuple[bool, list[str]]:
    """開けるか確認する。(開けたか, 中身に関するwarningのリスト) を返す。"""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            if zf.testzip() is not None:
                return False, []
            names = zf.namelist()
    except zipfile.BadZipFile:
        return False, []

    warnings = []
    if not any(n.startswith("XBRL/PublicDoc/") and n.endswith("_cal.xml") for n in names):
        warnings.append("XBRL/PublicDoc/*_cal.xml が見つかりません")
    if not any(n.startswith(_REQUIRED_AUDIT_DOC_PREFIX) for n in names):
        warnings.append("XBRL/AuditDoc/ が見つかりません")
    return True, warnings


def _default_confirm(message: str) -> bool:
    answer = input(f"{message}\n続行しますか？ [y/N]: ")
    return answer.strip().lower() == "y"


def _check_disk_space(
    raw_dir: Path,
    pending_count: int,
    estimated_bytes_per_doc: int,
    assume_yes: bool,
    confirm: Callable[[str], bool],
) -> None:
    probe_dir = raw_dir if raw_dir.exists() else raw_dir.parent
    usage = shutil.disk_usage(probe_dir)
    needed = pending_count * estimated_bytes_per_doc
    if needed <= usage.free:
        return
    message = (
        f"空き容量が不足する可能性があります: 見積り必要量 {needed / 1e9:.1f}GB"
        f" > 空き容量 {usage.free / 1e9:.1f}GB"
        f"({pending_count}件 x {estimated_bytes_per_doc / 1e6:.0f}MB/件で概算)"
    )
    log.warning(message)
    if assume_yes:
        return
    if not sys.stdin.isatty():
        log.warning(
            "stdin is not a tty (non-interactive run); proceeding without confirmation"
        )
        return
    if not confirm(message):
        raise RuntimeError("disk space check aborted by user")


def _fetch_one(client: EdinetClient, doc_id: str, documents_dir: Path) -> FetchLogEntry:
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        result = client.get_document(doc_id, type_=1)
    except EdinetApiError as exc:
        log.error("failed to fetch %s: %s", doc_id, exc)
        return FetchLogEntry(doc_id, "failed", exc.http_status, None, None, fetched_at, str(exc))

    ok, warnings = _validate_zip(result.content)
    if not ok:
        log.error("downloaded zip for %s is corrupt, discarding", doc_id)
        return FetchLogEntry(
            doc_id, "failed", 200, len(result.content), None, fetched_at, "corrupt zip"
        )
    for warning in warnings:
        log.warning("%s: %s", doc_id, warning)

    zip_path = documents_dir / f"{doc_id}.zip"
    zip_path.write_bytes(result.content)
    digest = hashlib.sha256(result.content).hexdigest()
    return FetchLogEntry(doc_id, "ok", 200, len(result.content), digest, fetched_at, None)


def fetch_documents(
    client: EdinetClient,
    records: tuple[DocumentRecord, ...],
    raw_dir: Path,
    index_dir: Path,
    *,
    estimated_bytes_per_doc: int = 5_000_000,
    assume_yes: bool = False,
    confirm: Callable[[str], bool] = _default_confirm,
    show_progress: bool = True,
) -> dict[str, FetchLogEntry]:
    """`records` のZIPを冪等に取得する。再実行しても ok 済みの docID は skip する。"""
    documents_dir = raw_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    fetch_log_path = index_dir / "fetch_log.csv"
    log_entries = _read_fetch_log(fetch_log_path)

    pending: list[DocumentRecord] = []
    already_ok = 0
    for record in records:
        existing = log_entries.get(record.docID)
        zip_path = documents_dir / f"{record.docID}.zip"
        if existing is not None and existing.status == "ok" and zip_path.exists():
            already_ok += 1
            continue
        pending.append(record)

    if pending:
        _check_disk_space(raw_dir, len(pending), estimated_bytes_per_doc, assume_yes, confirm)

    counts = {"ok": 0, "failed": 0, "skipped": already_ok}
    with tqdm(total=len(pending), desc="fetch documents", disable=not show_progress) as bar:
        for record in pending:
            entry = _fetch_one(client, record.docID, documents_dir)
            log_entries[record.docID] = entry
            counts[entry.status] = counts.get(entry.status, 0) + 1
            bar.set_postfix(counts)
            bar.update(1)
            _write_fetch_log(fetch_log_path, log_entries)

    log.info(
        "fetch complete: ok=%d failed=%d skipped=%d (total=%d)",
        counts["ok"],
        counts["failed"],
        counts["skipped"],
        len(records),
    )
    return log_entries
