"""EDINET API v2 の薄い同期クライアント。

書類一覧 API (``documents.json``) と書類取得 API (``documents/<docID>``) を
叩くだけの層。絞り込み・保存・冪等性などのオーケストレーションは
``document_list.py`` の責務であり、ここには入れない。

仕様書(2026年6月版, EDINET API 仕様書 Version 2)で確認した重要な挙動:

- 書類一覧 API は、パラメータ誤り等のエラー(400/404/500)でも
  **HTTPステータスは200のまま**で、本文の ``metadata.status`` に実際の
  ステータスが入る。429/401 はゲートウェイ層で弾かれるらしく、
  実際の HTTP ステータスも 429/401 になる(仕様書 3-3節)。
  そのため成功判定は HTTP ステータスと本文の両方を見る必要がある。
- 書類取得 API は常に HTTP 200 を返し、成功/失敗は
  ``Content-Type`` ヘッダ(``application/octet-stream`` か
  ``application/json`` か)で判定する(仕様書 3-2-2節)。
- ``Subscription-Key`` は HTTP ヘッダではなく **クエリパラメータ**。
  そのため httpx 自身がデフォルトで INFO ログに出す "HTTP Request: GET <url>"
  にキーがそのまま乗ってしまう。``httpx``/``httpcore`` ロガーにキーを
  マスクするフィルタを付けて対処している(``_RedactApiKeyFilter``)。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import httpx

log = logging.getLogger(__name__)

_RETRYABLE_5XX_PREFIX = "5"
_RETRYABLE_STATUS = {"429"}


class _RedactApiKeyFilter(logging.Filter):
    """httpx/httpcore が INFO ログに出すリクエストURLから API キーを隠す。"""

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key

    def filter(self, record: logging.LogRecord) -> bool:
        if self._api_key:
            message = record.getMessage()
            if self._api_key in message:
                record.msg = message.replace(self._api_key, "***REDACTED***")
                record.args = ()
        return True


def _install_api_key_redaction(api_key: str) -> None:
    for logger_name in ("httpx", "httpcore"):
        logger = logging.getLogger(logger_name)
        already_installed = any(
            isinstance(f, _RedactApiKeyFilter) and f._api_key == api_key for f in logger.filters
        )
        if not already_installed:
            logger.addFilter(_RedactApiKeyFilter(api_key))


# httpx のデフォルトタイムアウト(5秒)は書類取得APIのZIPダウンロードには
# 短すぎ、実運用で ReadTimeout が発生した(2026-09-18の実行で確認)。
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


class EdinetApiError(RuntimeError):
    """EDINET API がエラーを返した(リトライ済み、またはリトライ対象外)。"""

    def __init__(self, status: str, message: str, http_status: int) -> None:
        super().__init__(f"EDINET API error status={status} http_status={http_status}: {message}")
        self.status = status
        self.message = message
        self.http_status = http_status


@dataclass(frozen=True)
class DocumentListResponse:
    metadata: dict[str, object]
    results: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class DocumentFetchResult:
    content: bytes
    content_type: str


def _is_retryable_status(status: str) -> bool:
    return status in _RETRYABLE_STATUS or status.startswith(_RETRYABLE_5XX_PREFIX)


def _embedded_error(body: object) -> tuple[str, str] | None:
    """JSON 本文からエラーの (status, message) を取り出す。エラーで無ければ None。

    書類一覧APIの通常エラー形式 ``{"metadata": {"status": ..., "message": ...}}``
    と、ゲートウェイ層のエラー形式 ``{"StatusCode": ..., "message": ...}`` の
    両方に対応する。
    """
    if not isinstance(body, dict):
        return None
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and "status" in metadata:
        status = str(metadata["status"])
        if status != "200":
            return status, str(metadata.get("message", ""))
        return None
    if "StatusCode" in body:
        return str(body["StatusCode"]), str(body.get("message", ""))
    return None


class EdinetClient:
    """EDINET API v2 の書類一覧・書類取得エンドポイントを叩く同期クライアント。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        min_interval_seconds: float = 1.0,
        max_retries: int = 5,
        backoff_base_seconds: float = 2.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        _install_api_key_redaction(api_key)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._min_interval_seconds = min_interval_seconds
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        # httpxのデフォルト(5秒)はZIPダウンロードには短すぎるため緩める。
        self._client = client if client is not None else httpx.Client(timeout=_DEFAULT_TIMEOUT)
        self._sleep = sleep
        self._last_request_at: float | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdinetClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _throttle(self) -> None:
        if self._last_request_at is None:
            self._last_request_at = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._min_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)
        self._last_request_at = time.monotonic()

    def _backoff_delay(self, attempt: int) -> float:
        return float(self._backoff_base_seconds * (2**attempt))

    def list_documents(self, target_date: date, type_: int = 2) -> DocumentListResponse:
        """書類一覧API。``type_=2`` で提出書類一覧+メタデータを取得する。"""
        params = {
            "date": target_date.isoformat(),
            "type": str(type_),
            "Subscription-Key": self._api_key,
        }
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._client.get(f"{self._base_url}/documents.json", params=params)
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    log.warning(
                        "documents.json request failed (%s), retrying (attempt %d/%d)",
                        exc,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise EdinetApiError("network_error", str(exc), 599) from exc

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self._max_retries:
                    log.warning(
                        "documents.json returned HTTP %s, retrying (attempt %d/%d)",
                        response.status_code,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise EdinetApiError(
                    str(response.status_code), "max retries exceeded", response.status_code
                )
            if response.status_code != 200:
                raise EdinetApiError(str(response.status_code), response.text, response.status_code)

            body = response.json()
            error = _embedded_error(body)
            if error is not None:
                status, message = error
                if _is_retryable_status(status) and attempt < self._max_retries:
                    log.warning(
                        "documents.json returned embedded status %s, retrying (attempt %d/%d)",
                        status,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise EdinetApiError(status, message, response.status_code)

            results = tuple(body.get("results", []) or [])
            return DocumentListResponse(metadata=body.get("metadata", {}), results=results)

        raise EdinetApiError("max_retries_exceeded", "unreachable", 599)

    def get_document(self, doc_id: str, type_: int = 1) -> DocumentFetchResult:
        """書類取得API。``type_=1`` で提出本文書+監査報告書+XBRLのZIPを取得する。"""
        params = {"type": str(type_), "Subscription-Key": self._api_key}
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._client.get(f"{self._base_url}/documents/{doc_id}", params=params)
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    log.warning(
                        "documents/%s request failed (%s), retrying (attempt %d/%d)",
                        doc_id,
                        exc,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise EdinetApiError("network_error", str(exc), 599) from exc
            content_type = response.headers.get("content-type", "")

            if content_type.startswith("application/json"):
                body = response.json()
                error = _embedded_error(body) or (str(response.status_code), response.text)
                status, message = error
                if _is_retryable_status(status) and attempt < self._max_retries:
                    log.warning(
                        "documents/%s returned status %s, retrying (attempt %d/%d)",
                        doc_id,
                        status,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise EdinetApiError(status, message, response.status_code)

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self._max_retries:
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise EdinetApiError(
                    str(response.status_code), "max retries exceeded", response.status_code
                )
            if response.status_code != 200:
                raise EdinetApiError(str(response.status_code), response.text, response.status_code)

            return DocumentFetchResult(content=response.content, content_type=content_type)

        raise EdinetApiError("max_retries_exceeded", "unreachable", 599)
