"""Phase 1 end-to-end pipeline: GDELT -> extraction -> Neo4j graph.

Usage:
    uv run python -m ingestion.run --limit 100
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from dedup.decide import find_merge_target
from dedup.matching import find_candidates
from dedup.merge import merge_into_existing
from extraction.llm import ExtractionError, QuotaExhaustedError
from extraction.pipeline import extract_with_retry
from graph.client import session as neo4j_session
from graph.writer import build_payload, write_events_batch
from ingestion.fetch_text import fetch_article_text
from ingestion.gdelt import fetch_articles, parse_seendate
from ingestion.store import Article, connect, log_extraction_failure, mark_extracted, unextracted_articles, upsert_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FETCH_CONCURRENCY = 10


def fetch_and_buffer(limit: int) -> None:
    """Pull article metadata from GDELT, fetch body text (concurrently), store in SQLite."""
    records = [r for r in fetch_articles(max_records=limit) if r.get("url")]
    logger.info("GDELT returned %d article records", len(records))

    texts: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=FETCH_CONCURRENCY) as executor:
        future_to_url = {executor.submit(fetch_article_text, r["url"]): r["url"] for r in records}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                texts[url] = future.result()
            except Exception as e:
                logger.error("unexpected error fetching %s: %s", url, e)
                texts[url] = None

    with connect() as conn:
        fetched = 0
        for record in records:
            url = record["url"]
            text = texts.get(url)
            if not text or len(text) < 200:
                logger.info("skipping %s (no extractable body text)", url)
                continue
            upsert_article(
                conn,
                Article(
                    url=url,
                    title=record.get("title", ""),
                    domain=record.get("domain"),
                    language=record.get("language"),
                    seendate=str(parse_seendate(record.get("seendate", ""))),
                    body_text=text,
                ),
            )
            fetched += 1
    logger.info("buffered %d articles with body text", fetched)


def extract_and_write(limit: int) -> None:
    """Run extraction over buffered, not-yet-extracted articles, dedup each
    against the graph (including events written earlier in this same run),
    and write to Neo4j one event at a time.

    Events are written individually rather than batched: dedup candidate
    matching needs to see events written earlier in this run, so a new event
    must be visible in Neo4j before the next article's candidate search runs.
    At our scale (~100 articles/run) the extra round trips are cheap relative
    to the Gemini calls that dominate runtime anyway.
    """
    with connect() as conn, neo4j_session() as s:
        articles = unextracted_articles(conn, limit=limit)
        logger.info("extracting %d articles", len(articles))

        succeeded = failed = merged = 0

        for article in articles:
            try:
                event, embedding = extract_with_retry(
                    article.title, article.body_text, article.url
                )
            except QuotaExhaustedError:
                # Not this article's fault -- leave it unmarked so it's picked
                # up on a future run, and stop now since every remaining call
                # in this run would fail identically.
                logger.warning(
                    "quota exhausted after %d succeeded / %d failed; stopping run early "
                    "(%d articles left for a future run)",
                    succeeded,
                    failed,
                    len(articles) - succeeded - failed,
                )
                break
            except ExtractionError as e:
                log_extraction_failure(conn, article.url, str(e), e.raw_response)
                mark_extracted(conn, article.url)  # genuinely bad content, don't retry forever
                failed += 1
                continue

            payload = build_payload(event, embedding, article.url, article.domain)
            new_entity_ids = {ent["id"] for ent in payload["entities"]}
            candidates = find_candidates(s, embedding, payload["date"], exclude_id=payload["id"])
            survivor_id = find_merge_target(
                candidates, new_entity_ids, event.title, event.summary, payload["date"]
            )

            if survivor_id:
                merge_into_existing(s, survivor_id, payload)
                logger.info("merged %s into existing event %s", article.url, survivor_id)
                merged += 1
            else:
                write_events_batch(s, [payload])

            mark_extracted(conn, article.url)
            succeeded += 1

        logger.info(
            "extraction done: %d succeeded (%d merged into existing events), %d failed",
            succeeded,
            merged,
            failed,
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="The Dot -- Phase 1 ingestion pipeline")
    parser.add_argument("--limit", type=int, default=100, help="max articles to pull from GDELT")
    args = parser.parse_args()

    start = time.time()
    fetch_and_buffer(args.limit)
    extract_and_write(args.limit)
    logger.info("pipeline run finished in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
