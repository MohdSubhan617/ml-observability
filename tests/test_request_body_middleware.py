"""Tests for bounded request-body handling at the ASGI boundary."""

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest
from starlette.types import Message, Scope

from ml_observability.app import RequestBodyLimitMiddleware


def invoke_middleware(
    *,
    headers: list[tuple[bytes, bytes]],
    request_messages: Iterable[Message],
    max_bytes: int = 5,
) -> tuple[int, int, bytes | None]:
    """Run middleware against a controlled ASGI request and record consumed chunks."""
    messages = iter(request_messages)
    receive_count = 0
    status_code = 0
    response_body: bytes | None = None

    async def receive() -> Message:
        nonlocal receive_count
        receive_count += 1
        return next(messages)

    async def send(message: Message) -> None:
        nonlocal status_code, response_body
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_body = message.get("body", b"")

    async def downstream(scope: Scope, body_receive: Any, body_send: Any) -> None:
        while True:
            message = await body_receive()
            if not message.get("more_body", False):
                break
        await body_send({"type": "http.response.start", "status": 200, "headers": []})
        await body_send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestBodyLimitMiddleware(downstream, max_bytes=max_bytes)
    scope: Scope = {"type": "http", "method": "POST", "headers": headers}
    asyncio.run(middleware(scope, receive, send))
    return status_code, receive_count, response_body


def request_chunks(*chunks: bytes) -> list[Message]:
    """Build ASGI body messages with one chunk marked as the end."""
    return [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]


def test_accepts_exact_content_length_and_body_size() -> None:
    status, receive_count, body = invoke_middleware(
        headers=[(b"content-length", b"5")], request_messages=request_chunks(b"ab", b"cde")
    )

    assert (status, receive_count, body) == (200, 2, b"ok")


def test_accepts_body_without_content_length() -> None:
    status, receive_count, body = invoke_middleware(
        headers=[], request_messages=request_chunks(b"abcde")
    )

    assert (status, receive_count, body) == (200, 1, b"ok")


@pytest.mark.parametrize("header_value", [b"five", b"+5", b"-1", b""])
def test_rejects_malformed_content_length(header_value: bytes) -> None:
    status, receive_count, body = invoke_middleware(
        headers=[(b"content-length", header_value)], request_messages=request_chunks(b"abc")
    )

    assert status == 400
    assert receive_count == 0
    assert body == b'{"detail":"Invalid Content-Length header."}'


def test_rejects_duplicate_content_length_headers() -> None:
    status, receive_count, _ = invoke_middleware(
        headers=[(b"content-length", b"2"), (b"content-length", b"2")],
        request_messages=request_chunks(b"ab"),
    )

    assert status == 400
    assert receive_count == 0


def test_rejects_content_length_combined_with_transfer_encoding() -> None:
    status, receive_count, _ = invoke_middleware(
        headers=[(b"content-length", b"2"), (b"transfer-encoding", b"chunked")],
        request_messages=request_chunks(b"ab"),
    )

    assert status == 400
    assert receive_count == 0


@pytest.mark.parametrize(
    ("content_length", "chunks"),
    [(b"2", (b"abc",)), (b"4", (b"ab",))],
)
def test_rejects_content_length_that_does_not_match_body(
    content_length: bytes, chunks: tuple[bytes, ...]
) -> None:
    status, _, body = invoke_middleware(
        headers=[(b"content-length", content_length)], request_messages=request_chunks(*chunks)
    )

    assert status == 400
    assert body == b'{"detail":"Request body length does not match Content-Length."}'


def test_actual_body_limit_catches_misleading_small_content_length() -> None:
    status, receive_count, body = invoke_middleware(
        headers=[(b"content-length", b"2")],
        request_messages=request_chunks(b"123", b"456", b"unused"),
    )

    assert status == 413
    assert receive_count == 2
    assert body == b'{"detail":"Request body exceeds the configured size limit."}'


def test_chunked_body_is_rejected_before_remaining_chunks_are_read() -> None:
    status, receive_count, _ = invoke_middleware(
        headers=[(b"transfer-encoding", b"chunked")],
        request_messages=request_chunks(b"123", b"456", b"unused"),
    )

    assert status == 413
    assert receive_count == 2


def test_oversized_declared_content_length_is_rejected_without_reading_body() -> None:
    status, receive_count, _ = invoke_middleware(
        headers=[(b"content-length", b"6")], request_messages=request_chunks(b"unused")
    )

    assert status == 413
    assert receive_count == 0


@pytest.mark.parametrize("value", ["", "invalid", "0", "-1"])
def test_invalid_environment_limits_fail_with_clear_error(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("ML_OBSERVABILITY_MAX_REQUEST_BYTES", value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        RequestBodyLimitMiddleware(lambda *_args: None)
