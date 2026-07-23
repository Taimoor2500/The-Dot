"""Gemini client wrapper for event extraction and embeddings."""

import os

from google import genai
from google.genai import types
from pydantic import ValidationError

from extraction.schemas import ExtractedEvent

EXTRACTION_MODEL = os.environ.get("GEMINI_EXTRACTION_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768

SYSTEM_INSTRUCTION = """You extract structured events from a single news article.
Be neutral and factual. Only include entities and claims that are explicitly
supported by the article text. If the article does not describe a discrete
event, set confidence low (below 0.3)."""


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


def _client() -> genai.Client:
    # A fresh genai.Client() per call gets garbage-collected mid-request (its
    # httpx client closes itself in __del__), so cache a single instance.
    global _client_instance
    if _client_instance is None:
        _client_instance = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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
