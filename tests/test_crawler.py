"""Tests for the crawler: robots.txt, rate limiting, retries, storage.

Network access is replaced with ``httpx.MockTransport`` handlers so the
tests are hermetic and fast.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

import httpx
import pytest

import prometheus_ns.data.crawler as crawler_module
from prometheus_ns.data.crawler import (
    CrawlLog,
    DomainBucket,
    RobotsCache,
    fetch_one,
    gutenberg_urls,
)


def _handler_factory(robots_denied: bool = False, fail_twice: bool = False):
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls[url] = calls.get(url, 0) + 1
        if url.endswith("/robots.txt"):
            body = "User-agent: *\nDisallow: /private\n" if robots_denied else "User-agent: *\nAllow: /\n"
            return httpx.Response(200, text=body)
        if "/private/" in url:
            return httpx.Response(200, text="secret page")
        if fail_twice and calls[url] <= 2:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, text=f"<html><body>content of {url}</body></html>")

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_gutenberg_urls_are_deterministic() -> None:
    assert gutenberg_urls(42, 5) == gutenberg_urls(42, 5)
    assert len(gutenberg_urls(42, 5)) == 5
    assert all(u.startswith("https://www.gutenberg.org/") for u in gutenberg_urls(42, 5))


def test_robots_cache_allows_and_denies() -> None:
    async def run() -> None:
        async with _make_client(_handler_factory(robots_denied=True)) as client:
            robots = RobotsCache(client, "PrometheusNS-Research/0.1")
            assert await robots.allowed("https://example.org/public/page") is True
            assert await robots.allowed("https://example.org/private/page") is False

    asyncio.run(run())


def test_domain_bucket_enforces_spacing() -> None:
    async def run() -> None:
        bucket = DomainBucket(rps=200.0)  # 5 ms interval
        import time

        await bucket.wait("example.org")
        t0 = time.monotonic()
        await bucket.wait("example.org")
        await bucket.wait("example.org")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.005  # at least one interval was enforced

    asyncio.run(run())


def test_fetch_one_saves_file_and_logs(tmp_path) -> None:
    async def run() -> None:
        handler = _handler_factory()
        async with _make_client(handler) as client:
            bucket = DomainBucket(rps=1000.0)
            robots = RobotsCache(client, "PrometheusNS-Research/0.1")
            log = CrawlLog(tmp_path / "logs" / "crawl.jsonl")
            raw_dir = tmp_path / "data_raw"
            result = await fetch_one(
                client, bucket, robots, log, raw_dir,
                "https://example.org/books/one", timeout_s=5, retries=3,
            )
            assert result.status == 200 and result.path is not None
            saved = result.path.read_text(encoding="utf-8")
            assert "content of" in saved
            # Path layout: data_raw/{domain}/{sha256[:12]}.raw
            assert result.path.parent.name == "example.org"
            rows = [json.loads(line) for line in (tmp_path / "logs" / "crawl.jsonl").read_text().splitlines()]
            assert rows[-1]["url"] == "https://example.org/books/one"
            assert rows[-1]["status"] == 200
            assert rows[-1]["bytes"] == len(saved.encode("utf-8"))

    asyncio.run(run())


def test_fetch_one_retries_on_server_error(tmp_path, monkeypatch) -> None:
    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(crawler_module.asyncio, "sleep", instant_sleep)

    async def run() -> None:
        handler = _handler_factory(fail_twice=True)
        async with _make_client(handler) as client:
            result = await fetch_one(
                client, DomainBucket(rps=1000.0), RobotsCache(client, "t"),
                CrawlLog(tmp_path / "crawl.jsonl"), tmp_path / "raw",
                "https://retry.org/page", timeout_s=5, retries=3,
            )
            assert result.status == 200
            assert handler.calls["https://retry.org/page"] == 3

    asyncio.run(run())


def test_fetch_one_respects_robots(tmp_path) -> None:
    async def run() -> None:
        handler = _handler_factory(robots_denied=True)
        async with _make_client(handler) as client:
            result = await fetch_one(
                client, DomainBucket(rps=1000.0), RobotsCache(client, "t"),
                CrawlLog(tmp_path / "crawl.jsonl"), tmp_path / "raw",
                "https://example.org/private/thing", timeout_s=5, retries=3,
            )
            assert result.status == -1
            assert result.reason == "robots_disallowed"
            rows = [json.loads(line) for line in (tmp_path / "crawl.jsonl").read_text().splitlines()]
            assert rows[0]["status"] == -1

    asyncio.run(run())


def test_fetch_one_gives_up_after_retries(tmp_path, monkeypatch) -> None:
    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(crawler_module.asyncio, "sleep", instant_sleep)

    class AlwaysDown(httpx.AsyncClient):
        pass

    async def run() -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="down")

        async with _make_client(handler) as client:
            result = await fetch_one(
                client, DomainBucket(rps=1000.0), RobotsCache(client, "t"),
                CrawlLog(tmp_path / "crawl.jsonl"), tmp_path / "raw",
                "https://down.org/page", timeout_s=5, retries=3,
            )
            assert result.status == 500
            assert result.path is None

    asyncio.run(run())


def test_bucket_rejects_nonpositive_rps() -> None:
    with pytest.raises(ValueError):
        DomainBucket(rps=0.0)
