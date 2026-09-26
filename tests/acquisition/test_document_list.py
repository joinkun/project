"""document_list.py のテスト。実APIは叩かず httpx.MockTransport でモックする。"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from fsgen.acquisition.document_list import (
    DocumentRecord,
    fetch_document_list,
    fetch_documents,
    iter_dates,
    write_document_list_csv,
)
from fsgen.acquisition.edinet_client import EdinetClient

_API_KEY = "TEST-KEY"
_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
_DOC_TYPE_CODES = ("120",)
_ORDINANCE_CODES = ("010",)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> EdinetClient:
    transport = httpx.MockTransport(handler)
    return EdinetClient(
        api_key=_API_KEY,
        base_url=_BASE_URL,
        min_interval_seconds=0.0,
        backoff_base_seconds=0.0,
        client=httpx.Client(transport=transport),
        sleep=lambda _seconds: None,
    )


def _list_body(results: list[dict[str, object]]) -> dict[str, object]:
    return {"metadata": {"status": "200", "message": "OK"}, "results": results}


def _row(
    doc_id: str,
    edinet_code: str = "E00001",
    doc_type_code: str = "120",
    ordinance_code: str = "010",
    xbrl_flag: str = "1",
    withdrawal_status: str = "0",
    period_start: str = "2023-04-01",
) -> dict[str, object]:
    return {
        "docID": doc_id,
        "edinetCode": edinet_code,
        "secCode": "10000",
        "filerName": "テスト株式会社",
        "docTypeCode": doc_type_code,
        "ordinanceCode": ordinance_code,
        "periodStart": period_start,
        "periodEnd": "2024-03-31",
        "submitDateTime": "2024-06-20 10:00",
        "xbrlFlag": xbrl_flag,
        "withdrawalStatus": withdrawal_status,
        "docInfoEditStatus": "0",
    }


def test_iter_dates_inclusive() -> None:
    dates = list(iter_dates(dt.date(2024, 4, 1), dt.date(2024, 4, 3)))
    assert dates == [dt.date(2024, 4, 1), dt.date(2024, 4, 2), dt.date(2024, 4, 3)]


def test_iter_dates_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="must not be after"):
        list(iter_dates(dt.date(2024, 4, 3), dt.date(2024, 4, 1)))


def test_fetch_document_list_filters_by_doc_type_ordinance_xbrl_and_withdrawal() -> None:
    rows = [
        _row("S1000001"),
        _row("S1000002", doc_type_code="130"),  # 訂正報告書は除外
        _row("S1000003", ordinance_code="030"),  # 対象外の府令
        _row("S1000004", xbrl_flag="0"),  # XBRLなし
        _row("S1000005", withdrawal_status="1"),  # 取下げ
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_list_body(rows))

    with _client(handler) as client:
        summary = fetch_document_list(
            client,
            dt.date(2024, 4, 1),
            dt.date(2024, 4, 1),
            _DOC_TYPE_CODES,
            _ORDINANCE_CODES,
            require_xbrl=True,
            show_progress=False,
        )

    assert [r.docID for r in summary.records] == ["S1000001"]
    assert summary.raw_result_count == 5


def test_fetch_document_list_dedupes_by_docid_keeping_latest_seen() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            row = _row("S1000001", period_start="2023-04-01")
        else:
            row = _row("S1000001", period_start="2023-05-02")
        return httpx.Response(200, json=_list_body([row]))

    with _client(handler) as client:
        summary = fetch_document_list(
            client,
            dt.date(2024, 6, 12),
            dt.date(2024, 6, 13),
            _DOC_TYPE_CODES,
            _ORDINANCE_CODES,
            require_xbrl=True,
            show_progress=False,
        )

    assert summary.unique_document_count == 1
    assert summary.records[0].periodStart == "2023-05-02"


def test_fetch_document_list_reports_duplicate_edinet_codes() -> None:
    rows = [_row("S1000001", edinet_code="E00001"), _row("S1000002", edinet_code="E00001")]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_list_body(rows))

    with _client(handler) as client:
        summary = fetch_document_list(
            client,
            dt.date(2024, 4, 1),
            dt.date(2024, 4, 1),
            _DOC_TYPE_CODES,
            _ORDINANCE_CODES,
            require_xbrl=True,
            show_progress=False,
        )

    assert summary.duplicate_edinet_codes == {"E00001": 2}


def test_fetch_document_list_continues_past_a_failed_date() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "2024-04-01":
            return httpx.Response(200, json={"metadata": {"status": "500", "message": "err"}})
        return httpx.Response(200, json=_list_body([_row("S1000002")]))

    with _client(handler) as client:
        summary = fetch_document_list(
            client,
            dt.date(2024, 4, 1),
            dt.date(2024, 4, 2),
            _DOC_TYPE_CODES,
            _ORDINANCE_CODES,
            require_xbrl=True,
            show_progress=False,
        )

    assert summary.failed_dates == (dt.date(2024, 4, 1),)
    assert [r.docID for r in summary.records] == ["S1000002"]


def test_write_document_list_csv(tmp_path: Path) -> None:
    record = DocumentRecord.from_api_row(_row("S1000001"))
    from fsgen.acquisition.document_list import DocumentListSummary

    summary = DocumentListSummary(
        records=(record,),
        days_queried=1,
        raw_result_count=1,
        duplicate_edinet_codes={},
        failed_dates=(),
    )
    csv_path = tmp_path / "index" / "document_list.csv"
    write_document_list_csv(summary, csv_path)

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["docID"] == "S1000001"
    assert rows[0]["edinetCode"] == "E00001"


def _make_zip_bytes(*, with_cal: bool, with_audit_doc: bool) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        if with_cal:
            zf.writestr("XBRL/PublicDoc/sample_cal.xml", "<x/>")
        else:
            zf.writestr("XBRL/PublicDoc/readme.txt", "no cal file here")
        if with_audit_doc:
            zf.writestr("XBRL/AuditDoc/report.xbrl", "<x/>")
    return buffer.getvalue()


def test_fetch_documents_downloads_and_records_sha256(tmp_path: Path) -> None:
    zip_bytes = _make_zip_bytes(with_cal=True, with_audit_doc=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = tmp_path / "raw" / "index"

    with _client(handler) as client:
        entries = fetch_documents(
            client, (record,), raw_dir, index_dir, assume_yes=True, show_progress=False
        )

    assert entries["S1000001"].status == "ok"
    assert entries["S1000001"].sha256 == hashlib.sha256(zip_bytes).hexdigest()
    saved = (raw_dir / "documents" / "S1000001.zip").read_bytes()
    assert saved == zip_bytes

    log_rows = list(csv.DictReader((index_dir / "fetch_log.csv").open(encoding="utf-8")))
    assert log_rows[0]["status"] == "ok"


def test_fetch_documents_suffix_separates_output_dirs(tmp_path: Path) -> None:
    zip_bytes = _make_zip_bytes(with_cal=True, with_audit_doc=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = tmp_path / "raw" / "index"

    with _client(handler) as client:
        fetch_documents(
            client,
            (record,),
            raw_dir,
            index_dir,
            assume_yes=True,
            show_progress=False,
            suffix="_2023_2024",
        )

    assert (raw_dir / "documents_2023_2024" / "S1000001.zip").read_bytes() == zip_bytes
    assert (index_dir / "fetch_log_2023_2024.csv").exists()
    assert not (raw_dir / "documents").exists()
    assert not (index_dir / "fetch_log.csv").exists()


def test_fetch_documents_skips_already_ok_entries_without_http_call(tmp_path: Path) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200, content=b"unused", headers={"content-type": "application/octet-stream"}
        )

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = raw_dir / "index"
    documents_dir = raw_dir / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "S1000001.zip").write_bytes(b"already-here")
    index_dir.mkdir(parents=True)
    with (index_dir / "fetch_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["docID", "status", "http_status", "bytes", "sha256", "fetched_at", "error"]
        )
        writer.writerow(["S1000001", "ok", "200", "13", "deadbeef", "2024-06-20T10:00:00", ""])

    with _client(handler) as client:
        entries = fetch_documents(
            client, (record,), raw_dir, index_dir, assume_yes=True, show_progress=False
        )

    assert call_count == 0
    assert entries["S1000001"].status == "ok"
    assert entries["S1000001"].sha256 == "deadbeef"


def test_fetch_documents_discards_corrupt_zip(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"not a zip file", headers={"content-type": "application/octet-stream"}
        )

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = raw_dir / "index"

    with _client(handler) as client:
        entries = fetch_documents(
            client, (record,), raw_dir, index_dir, assume_yes=True, show_progress=False
        )

    assert entries["S1000001"].status == "failed"
    assert not (raw_dir / "documents" / "S1000001.zip").exists()


def test_fetch_documents_warns_but_keeps_zip_missing_audit_doc(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    zip_bytes = _make_zip_bytes(with_cal=True, with_audit_doc=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = raw_dir / "index"

    with caplog.at_level("WARNING"), _client(handler) as client:
        entries = fetch_documents(
            client, (record,), raw_dir, index_dir, assume_yes=True, show_progress=False
        )

    assert entries["S1000001"].status == "ok"
    assert (raw_dir / "documents" / "S1000001.zip").exists()
    assert any("AuditDoc" in record.getMessage() for record in caplog.records)


def test_fetch_documents_disk_space_check_prompts_and_aborts_on_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # プロンプトはインタラクティブ端末(stdin.isatty()==True)のときだけ出る。
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called: aborted before fetching")

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = raw_dir / "index"

    with _client(handler) as client, pytest.raises(RuntimeError, match="disk space"):
        fetch_documents(
            client,
            (record,),
            raw_dir,
            index_dir,
            estimated_bytes_per_doc=10**18,
            assume_yes=False,
            confirm=lambda _message: False,
            show_progress=False,
        )


def test_fetch_documents_disk_space_check_proceeds_on_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    zip_bytes = _make_zip_bytes(with_cal=True, with_audit_doc=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = raw_dir / "index"

    with _client(handler) as client:
        entries = fetch_documents(
            client,
            (record,),
            raw_dir,
            index_dir,
            estimated_bytes_per_doc=10**18,
            assume_yes=False,
            confirm=lambda _message: True,
            show_progress=False,
        )

    assert entries["S1000001"].status == "ok"


def test_fetch_documents_disk_space_check_skips_prompt_when_stdin_not_a_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # nohup / バックグラウンド実行では stdin が tty ではないため、input() で
    # ハングせずに warning を出して続行しなければならない。
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    zip_bytes = _make_zip_bytes(with_cal=True, with_audit_doc=True)
    confirm_calls = 0

    def confirm(_message: str) -> bool:
        nonlocal confirm_calls
        confirm_calls += 1
        return False  # 呼ばれたら fail させる目印

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    record = DocumentRecord.from_api_row(_row("S1000001"))
    raw_dir = tmp_path / "raw"
    index_dir = raw_dir / "index"

    with caplog.at_level("WARNING"), _client(handler) as client:
        entries = fetch_documents(
            client,
            (record,),
            raw_dir,
            index_dir,
            estimated_bytes_per_doc=10**18,
            assume_yes=False,
            confirm=confirm,
            show_progress=False,
        )

    assert confirm_calls == 0
    assert entries["S1000001"].status == "ok"
    assert any("non-interactive" in r.getMessage() for r in caplog.records)


def test_fetch_documents_does_not_stop_on_single_failure(tmp_path: Path) -> None:
    zip_bytes = _make_zip_bytes(with_cal=True, with_audit_doc=True)

    def handler(request: httpx.Request) -> httpx.Response:
        doc_id = request.url.path.rsplit("/", 1)[-1]
        if doc_id == "S1000001":
            return httpx.Response(404, json={"metadata": {"status": "404", "message": "Not Found"}})
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    records = (
        DocumentRecord.from_api_row(_row("S1000001")),
        DocumentRecord.from_api_row(_row("S1000002")),
    )
    raw_dir = tmp_path / "raw"
    index_dir = raw_dir / "index"

    with _client(handler) as client:
        entries = fetch_documents(
            client, records, raw_dir, index_dir, assume_yes=True, show_progress=False
        )

    assert entries["S1000001"].status == "failed"
    assert entries["S1000002"].status == "ok"
