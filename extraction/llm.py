"""Gemini client wrapper for event extraction and embeddings."""

import os

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from extraction.schemas import ExtractedEvent

EXTRACTION_MODEL = os.environ.get("GEMINI_EXTRACTION_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768

SYSTEM_INSTRUCTION = """You extract structured events from a single news article.
Be neutral and factual. Only include entities and claims that are explicitly
supported by the article text. If the article does not describe a discrete
event, set confidence low (below 0.3)."""

ADJUDICATION_SYSTEM = """You judge whether two event descriptions refer to the same
specific real-world event, not just a similar topic. Two articles covering the same
summit, decision, incident, or announcement are the same event even if worded very
differently. Two articles about a similar but distinct occurrence (different dates,
different specific incidents, or only a shared general topic) are NOT the same event."""


class AdjudicationResult(BaseModel):
    same_event: bool
    rationale: str


class ExtractionError(Exception):
    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


class QuotaExhaustedError(Exception):
    """Raised when the API rejects a call due to daily/rate quota exhaustion.

    Distinct from ExtractionError: this means "try this article again later",
    not "this article's content is bad" -- callers must not mark the article
    as permanently done.
    """


_client_instance: genai.Client | None = None


REQUEST_TIMEOUT_MS = 60_000  # google-genai passes timeout=None to httpx by default,
# which means "wait forever" -- a stalled connection hangs the whole run with no
# error and no log line. Bound every call explicitly instead.


def _client() -> genai.Client:
    # A fresh genai.Client() per call gets garbage-collected mid-request (its
    # httpx client closes itself in __del__), so cache a single instance.
    global _client_instance
    if _client_instance is None:
        _client_instance = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
    return _client_instance


def extract_event(article_title: str, article_text: str, source_url: str) -> ExtractedEvent:
    """Call Gemini once to extract a structured event from an article.

    Raises ExtractionError if the model output fails schema validation.
    Callers are responsible for retrying.
    """
    prompt = (
        f"Article title: {article_title}\n"
        f"Article URL: {source_url}\n\n"
        f"Article text:\n{article_text[:8000]}"
    )

    response = _client().models.generate_content(
        model=EXTRACTION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ExtractedEvent,
            temperature=0.1,
        ),
    )

    raw = response.text
    if raw is None:
        raise ExtractionError("empty response from model", raw_response=None)

    try:
        return ExtractedEvent.model_validate_json(raw)
    except ValidationError as e:
        raise ExtractionError(str(e), raw_response=raw) from e


def adjudicate_same_event(
    a_title: str,
    a_summary: str,
    a_date: str | None,
    b_title: str,
    b_summary: str,
    b_date: str | None,
) -> AdjudicationResult:
    """Cheap LLM yes/no call for borderline dedup candidates.

    Raises ExtractionError if the response fails schema validation. Callers
    should treat that as "assume not the same event" -- a missed merge is
    recoverable later, an incorrect merge corrupts the graph.
    """
    prompt = (
        f"Event A: {a_title} -- {a_summary} (date: {a_date})\n"
        f"Event B: {b_title} -- {b_summary} (date: {b_date})\n\n"
        "Are these the same real-world event?"
    )

    response = _client().models.generate_content(
        model=EXTRACTION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=ADJUDICATION_SYSTEM,
            response_mime_type="application/json",
            response_schema=AdjudicationResult,
            temperature=0.0,
        ),
    )

    raw = response.text
    if raw is None:
        raise ExtractionError("empty response from adjudication call", raw_response=None)

    try:
        return AdjudicationResult.model_validate_json(raw)
    except ValidationError as e:
        raise ExtractionError(str(e), raw_response=raw) from e


def embed_text(text: str) -> list[float]:
    """Embed a single string (e.g. `title + summary`) for vector search."""
    response = _client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values
