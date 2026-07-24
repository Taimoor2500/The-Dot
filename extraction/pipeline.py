"""Extraction pipeline: article text -> validated ExtractedEvent + embedding."""

import logging

import httpx
from google.genai.errors import APIError

from extraction.llm import ExtractionError, QuotaExhaustedError, embed_text, extract_event
from extraction.schemas import ExtractedEvent

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3  # 1 initial call + 2 retries


def _is_quota_exhausted(e: APIError) -> bool:
    return e.code == 429 and e.status == "RESOURCE_EXHAUSTED"


def extract_with_retry(
    article_title: str, article_text: str, source_url: str
) -> tuple[ExtractedEvent, list[float]]:
    """Extract a structured event and its embedding, retrying on validation failure.

    Raises ExtractionError (with the last raw response attached) if all
    attempts fail -- callers should log this to the failures table rather
    than dropping the article silently.

    Raises QuotaExhaustedError immediately, without retrying, if the API
    rejects the call for daily/rate quota reasons -- retrying a quota error
    within the same run wastes time and attempts for no benefit, and callers
    must leave the article unmarked so it's retried on a future run instead
    of a real extraction failure.
    """
    last_error: ExtractionError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            event = extract_event(article_title, article_text, source_url)
            embedding = embed_text(f"{event.title}\n{event.summary}")
            return event, embedding
        except ExtractionError as e:
            last_error = e
        except APIError as e:
            if _is_quota_exhausted(e):
                raise QuotaExhaustedError(str(e)) from e
            # Covers other embedding/extraction API failures (transient server
            # errors, etc.) that aren't schema validation errors but should
            # still be retried and logged rather than crashing the run.
            last_error = ExtractionError(str(e), raw_response=None)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            # google-genai's default retry_options is None (no SDK-level retry,
            # reraise=True), so these come straight through uncaught otherwise.
            last_error = ExtractionError(f"network error: {e}", raw_response=None)

        logger.warning(
            "extraction attempt %d/%d failed for %s: %s",
            attempt,
            MAX_ATTEMPTS,
            source_url,
            last_error,
        )

    assert last_error is not None
    raise last_error
