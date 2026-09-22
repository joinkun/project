"""EdinetClient のテスト。実APIは叩かず httpx.MockTransport でモックする。"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable

import httpx
import pytest

from fsgen.acquisition.edinet_client import DocumentFetchResult, EdinetApiError, EdinetClient

_API_KEY = "TEST-SECRET-KEY"
_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_retries: int = 5,
) -> tuple[EdinetClient, list[float]]:
    transport = httpx.MockTransport(handler)
    sleeps: list[float] = []
    client = EdinetClient(
        api_key=_API_KEY,
        base_url=_BASE_URL,
        max_retries=max_retries,
        backoff_base_seconds=0.01,
        min_interval_seconds=0.0,
        client=httpx.Client(transport=transport),
        sleep=sleeps.append,
    )
    return client, sleeps


def _list_body(
    status: str = "200", message: str = "OK", results: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "metadata": {
            "title": "提出された書類を把握するための API",
            "status": status,
            "message": message,
        },
        "results": results or [],
    }


def test_list_documents_success_passes_subscription_key_as_query_param() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_list_body(results=[{"docID": "S1000001"}]))

    client, _ = _make_client(handler)
    response = client.list_documents(dt.date(2024, 4, 1))

    assert response.results == ({"docID": "S1000001"},)
    assert len(captured) == 1
    assert captured[0].url.params["Subscription-Key"] == _API_KEY
    assert captured[0].url.params["date"] == "2024-04-01"
    assert captured[0].url.params["type"] == "2"


def test_list_documents_embedded_error_with_http_200_raises_without_retry() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=_list_body(status="404", message="Not Found"))

    client, sleeps = _make_client(handler)
    with pytest.raises(EdinetApiError) as excinfo:
        client.list_documents(dt.date(2024, 4, 1))

    assert excinfo.value.status == "404"
    assert call_count == 1
    assert sleeps == []


def test_list_documents_retries_on_http_429_then_succeeds() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429, json={"StatusCode": 429, "message": "Too Many Requests"})
        return httpx.Response(200, json=_list_body(results=[{"docID": "S1000001"}]))

    client, sleeps = _make_client(handler)
    response = client.list_documents(dt.date(2024, 4, 1))

    assert call_count == 3
    assert len(sleeps) == 2
    assert response.results == ({"docID": "S1000001"},)


def test_list_documents_retries_on_embedded_500_then_succeeds() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200, json=_list_body(status="500", message="Internal Server Error")
            )
        return httpx.Response(200, json=_list_body(results=[{"docID": "S1000001"}]))

    client, sleeps = _make_client(handler)
    response = client.list_documents(dt.date(2024, 4, 1))

    assert call_count == 2
    assert len(sleeps) == 1
    assert response.results == ({"docID": "S1000001"},)


def test_list_documents_gives_up_after_max_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client, sleeps = _make_client(handler, max_retries=2)
    with pytest.raises(EdinetApiError):
        client.list_documents(dt.date(2024, 4, 1))


def test_list_documents_retries_on_read_timeout_then_succeeds() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=_list_body(results=[{"docID": "S1000001"}]))

    client, sleeps = _make_client(handler)
    response = client.list_documents(dt.date(2024, 4, 1))

    assert call_count == 2
    assert len(sleeps) == 1
    assert response.results == ({"docID": "S1000001"},)


def test_get_document_retries_on_read_timeout_then_succeeds() -> None:
    call_count = 0
    zip_bytes = b"PK\x03\x04fake-zip-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    client, sleeps = _make_client(handler)
    result = client.get_document("S1000001")

    assert call_count == 2
    assert len(sleeps) == 1
    assert result.content == zip_bytes


def test_get_document_gives_up_after_max_retries_on_read_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, sleeps = _make_client(handler, max_retries=2)
    with pytest.raises(EdinetApiError) as excinfo:
        client.get_document("S1000001")

    assert excinfo.value.status == "network_error"
    assert len(sleeps) == 2

    assert len(sleeps) == 2


def test_get_document_success_returns_bytes_and_content_type() -> None:
    zip_bytes = b"PK\x03\x04fake-zip-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["type"] == "1"
        assert request.url.params["Subscription-Key"] == _API_KEY
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    client, _ = _make_client(handler)
    result = client.get_document("S1000001")

    assert isinstance(result, DocumentFetchResult)
    assert result.content == zip_bytes
    assert result.content_type == "application/octet-stream"


def test_get_document_json_error_401_raises_without_retry() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = {
            "StatusCode": 401,
            "message": "Access denied due to invalid subscription key.",
        }
        return httpx.Response(
            401,
            content=json.dumps(body),
            headers={"content-type": "application/json; charset=utf-8"},
        )

    client, sleeps = _make_client(handler)
    with pytest.raises(EdinetApiError) as excinfo:
        client.get_document("S1000001")

    assert excinfo.value.status == "401"
    assert call_count == 1
    assert sleeps == []


def test_get_document_retries_on_json_429_then_succeeds() -> None:
    call_count = 0
    zip_bytes = b"PK\x03\x04fake-zip-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            body = {"StatusCode": 429, "message": "Too Many Requests"}
            return httpx.Response(
                429, content=json.dumps(body), headers={"content-type": "application/json"}
            )
        return httpx.Response(
            200, content=zip_bytes, headers={"content-type": "application/octet-stream"}
        )

    client, sleeps = _make_client(handler)
    result = client.get_document("S1000001")

    assert call_count == 2
    assert len(sleeps) == 1
    assert result.content == zip_bytes


def test_api_key_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, json={"StatusCode": 429, "message": "Too Many Requests"})
        return httpx.Response(200, json=_list_body(results=[]))

    client, _ = _make_client(handler)
    with caplog.at_level("DEBUG"):
        client.list_documents(dt.date(2024, 4, 1))

    for record in caplog.records:
        assert _API_KEY not in record.getMessage()


def test_throttle_sleeps_between_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_list_body(results=[]))

    transport = httpx.MockTransport(handler)
    sleeps: list[float] = []
    client = EdinetClient(
        api_key=_API_KEY,
        base_url=_BASE_URL,
        min_interval_seconds=1.0,
        client=httpx.Client(transport=transport),
        sleep=sleeps.append,
    )

    client.list_documents(dt.date(2024, 4, 1))
    client.list_documents(dt.date(2024, 4, 2))

    assert len(sleeps) == 1
    assert sleeps[0] > 0.9
