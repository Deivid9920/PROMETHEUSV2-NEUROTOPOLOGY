"""Autonomous crawler with per-domain rate limiting and robots.txt checks.

Discovers candidate URLs from the configured sources, fetches them with
``httpx.AsyncClient`` under a token-bucket of ``data.rps`` requests per
second per domain, and stores raw responses under ``data_raw/{domain}/``
named by the first 12 hex chars of the URL SHA-256. Every attempt is
appended to ``logs/crawl.jsonl`` so later stages can audit what was
downloaded.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from prometheus_ns import cfg_get, ensure_dir, load_config, repo_path

_GUTENBERG_ID_POOL = range(1000, 40000)
_ARXIV_QUERY = 'cat:cs.CL AND cat:cs.LG'
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/html"


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one URL fetch attempt."""

    url: str
    status: int
    bytes_written: int
    path: Path | None
    reason: str | None = None


class CrawlLog:
    """Append-only JSONL registry of fetch attempts."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, record: dict) -> None:
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class RobotsCache:
    """Per-domain robots.txt parser with a fail-open network policy.

    A missing or unreachable robots.txt allows crawling; an explicit
    ``Disallow`` for the configured User-Agent denies it.
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._cache:
            parser = urllib.robotparser.RobotFileParser()
            try:
                resp = await self._client.get(f"{domain}/robots.txt")
                if resp.status_code in (401, 403):
                    parser.parse(["User-agent: *", "Disallow: /"])
                else:
                    parser.parse(resp.text.splitlines())
                self._cache[domain] = parser
            except httpx.HTTPError:
                self._cache[domain] = None
        parser = self._cache[domain]
        return True if parser is None else parser.can_fetch(self._user_agent, url)


class DomainBucket:
    """Token bucket enforcing a minimum interval between requests per domain."""

    def __init__(self, rps: float) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")
        self._interval = 1.0 / rps
        self._last: dict[str, float] = {}

    async def wait(self, domain: str) -> None:
        now = time.monotonic()
        elapsed = now - self._last.get(domain, 0.0)
        if elapsed < self._interval:
            await asyncio.sleep(self._interval - elapsed)
        self._last[domain] = time.monotonic()


def gutenberg_urls(seed: int, count: int) -> list[str]:
    """Deterministic sample of Project Gutenberg plain-text book URLs."""
    rng = random.Random(seed)
    ids = rng.sample(list(_GUTENBERG_ID_POOL), k=min(count, len(_GUTENBERG_ID_POOL)))
    return [f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt" for gid in ids]


def arxiv_urls(seed: int, count: int, user_agent: str = "PrometheusNS-Research/0.1") -> list[str]:
    """Discover arXiv abstract pages through the public export API.

    Discovery uses ``urllib`` because the export API answers httpx
    requests with 406 regardless of headers; the identified crawler
    User-Agent is sent on every call.
    """
    import urllib.error
    import urllib.request

    rng = random.Random(seed)
    start = rng.randrange(0, 2000)
    urls: list[str] = []
    batch = 20
    while len(urls) < count:
        api = (
            "https://export.arxiv.org/api/query"
            f"?search_query={_ARXIV_QUERY}&start={start}&max_results={batch}"
        )
        try:
            request = urllib.request.Request(
                api.replace(" ", "%20"), headers={"User-Agent": user_agent}
            )
            with urllib.request.urlopen(request, timeout=20.0) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError):
            break
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("<id>http://arxiv.org/abs/"):
                abs_url = line.removeprefix("<id>").removesuffix("</id>")
                urls.append(abs_url.replace("http://", "https://"))
        start += batch
    return urls[:count]


async def wikipedia_urls(client: httpx.AsyncClient, seed: int, count: int) -> list[str]:
    """Discover article URLs through the MediaWiki random-title API."""
    titles: list[str] = []
    rng = random.Random(seed)
    # rnlocale-independent parameters; the API returns namespace-0 titles.
    while len(titles) < count:
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": "0",
            "rnlimit": "20",
            "format": "json",
        }
        try:
            resp = await client.get(_WIKI_API, params=params)
        except httpx.HTTPError:
            break
        if resp.status_code != 200:
            break
        payload = resp.json()
        batch_titles = [item["title"] for item in payload.get("query", {}).get("random", [])]
        if not batch_titles:
            break
        titles.extend(batch_titles)
    return [f"{_WIKI_REST}/{t.replace(' ', '_')}" for t in titles[:count]]


async def discover_urls(cfg: dict, client: httpx.AsyncClient) -> list[tuple[str, str]]:
    """Return a seeded, shuffled list of (source, url) pairs to fetch."""
    seed = int(cfg_get(cfg, "seed", 42))
    per_source = int(cfg_get(cfg, "data.max_pages_per_source", 200))
    pairs: list[tuple[str, str]] = []
    sources = cfg_get(cfg, "data.sources", ["gutenberg", "arxiv", "wikipedia"])
    for source in sources:
        if source == "gutenberg":
            pairs.extend(("gutenberg", u) for u in gutenberg_urls(seed, per_source))
        elif source == "arxiv":
            ua = str(cfg_get(cfg, "data.user_agent", "PrometheusNS-Research/0.1"))
            pairs.extend(("arxiv", u) for u in arxiv_urls(seed, per_source, user_agent=ua))
        elif source == "wikipedia":
            titles = await wikipedia_urls(client, seed, per_source)
            pairs.extend(("wikipedia", u) for u in titles)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


async def fetch_one(
    client: httpx.AsyncClient,
    bucket: DomainBucket,
    robots: RobotsCache,
    crawl_log: CrawlLog,
    raw_dir: Path,
    url: str,
    timeout_s: float,
    retries: int,
) -> FetchResult:
    """Fetch a single URL with robots check, rate limiting and retries.

    Retries use exponential backoff of 2/4/8 seconds. Successful HTML or
    text responses are stored under ``raw_dir/{domain}/{sha256[:12]}``.
    """
    domain = urlparse(url).netloc
    if not await robots.allowed(url):
        await crawl_log.append({"url": url, "status": -1, "bytes": 0, "ts": time.time(), "note": "robots_disallowed"})
        return FetchResult(url=url, status=-1, bytes_written=0, path=None, reason="robots_disallowed")

    backoffs = (2.0, 4.0, 8.0)
    last_status = -1
    for attempt in range(max(1, retries)):
        await bucket.wait(domain)
        try:
            resp = await client.get(url, timeout=timeout_s)
        except httpx.HTTPError as exc:
            last_status = -1
            await crawl_log.append({"url": url, "status": -1, "bytes": 0, "ts": time.time(), "note": f"error:{type(exc).__name__}"})
            if attempt < retries - 1:
                await asyncio.sleep(backoffs[min(attempt, len(backoffs) - 1)])
            continue
        last_status = resp.status_code
        if resp.status_code == 200 and resp.content:
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
            domain_dir = ensure_dir(raw_dir / domain.replace(":", "_"))
            target = domain_dir / f"{digest}.raw"
            target.write_bytes(resp.content)
            await crawl_log.append({"url": url, "status": 200, "bytes": len(resp.content), "ts": time.time()})
            return FetchResult(url=url, status=200, bytes_written=len(resp.content), path=target)
        await crawl_log.append({"url": url, "status": resp.status_code, "bytes": len(resp.content or b""), "ts": time.time()})
        if resp.status_code in (403, 429):
            break  # rate limited or blocked: retrying only worsens standing
        if attempt < retries - 1:
            await asyncio.sleep(backoffs[min(attempt, len(backoffs) - 1)])
    return FetchResult(url=url, status=last_status, bytes_written=0, path=None, reason="fetch_failed")


async def run_crawl(cfg: dict) -> list[FetchResult]:
    """Execute one crawl pass over every configured source."""
    raw_dir = repo_path(cfg, cfg_get(cfg, "data.out_dir", "data_raw"))
    crawl_log = CrawlLog(repo_path(cfg, "logs", "crawl.jsonl"))
    rps = float(cfg_get(cfg, "data.rps", 1.0))
    timeout_s = float(cfg_get(cfg, "data.timeout_s", 20))
    retries = int(cfg_get(cfg, "data.retries", 3))
    user_agent = str(cfg_get(cfg, "data.user_agent", "PrometheusNS-Research/0.1"))

    results: list[FetchResult] = []
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": user_agent}) as client:
        robots = RobotsCache(client, user_agent)
        bucket = DomainBucket(rps)
        pairs = await discover_urls(cfg, client)
        for source, url in pairs:
            # `source` only drives discovery; the crawl log records the
            # URL, from which the domain and origin are recoverable.
            results.append(await fetch_one(client, bucket, robots, crawl_log, raw_dir, url, timeout_s, retries))
    return results


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.data.crawler --config PATH``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS autonomous crawler")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = asyncio.run(run_crawl(cfg))
    ok = sum(1 for r in results if r.status == 200)
    skipped = sum(1 for r in results if r.status == -1)
    print(f"crawl finished: {len(results)} attempted, {ok} fetched, {skipped} skipped, {len(results) - ok - skipped} failed")


if __name__ == "__main__":
    main()
