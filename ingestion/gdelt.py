"""Fetch articles from the GDELT 2.0 DOC API."""

from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Geopolitics & trade domain scope (Phase 1).
DEFAULT_QUERY = (
    '(sanctions OR tariff OR "trade war" OR embargo OR "export controls" '
    'OR diplomatic OR summit OR treaty) sourcelang:english'
)


class GdeltArticle(dict):
    """A single article record as returned by the GDELT DOC API ArtList mode."""


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=10, max=60))
def _fetch_page(client: httpx.Client, params: dict) -> dict:
    # GDELT's shared-IP rate limit can fluctuate within seconds; this is a
    # batch job (not latency-sensitive), so retry patiently rather than fail fast.
    response = client.get(GDELT_DOC_API, params=params)
    response.raise_for_status()
    return response.json()


def fetch_articles(
    query: str = DEFAULT_QUERY,
    max_records: int = 250,
    timespan: str = "60days",
) -> list[GdeltArticle]:
    """Query the GDELT DOC API and return raw article records.

    `timespan` uses GDELT's relative window syntax (e.g. "60days").
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, 250),  # GDELT DOC API page cap
        "timespan": timespan,
        "sort": "DateDesc",
    }
    with httpx.Client(timeout=30.0) as client:
        payload = _fetch_page(client, params)

    articles = payload.get("articles", [])
    return [GdeltArticle(a) for a in articles]


def parse_seendate(raw: str) -> datetime | None:
    """GDELT seendate looks like '20260722T153000Z'."""
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
