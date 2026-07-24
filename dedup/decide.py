"""Merge decision: cosine + entity-Jaccard threshold rule, with LLM adjudication
for borderline candidates.

Thresholds are placeholder defaults (documented in the plan as needing tuning
on a hand-labeled sample) -- run `evals/score_dedup.py` against gold set #1
and adjust via the DEDUP_* env vars once enough labeled pairs exist.
"""

import logging
import os
from typing import Literal

from google.genai.errors import APIError

from extraction.llm import ExtractionError, adjudicate_same_event
from dedup.jaccard import jaccard

logger = logging.getLogger(__name__)

COSINE_MATCH = float(os.environ.get("DEDUP_COSINE_MATCH", "0.90"))
COSINE_FLOOR = float(os.environ.get("DEDUP_COSINE_FLOOR", "0.70"))
JACCARD_MATCH = float(os.environ.get("DEDUP_JACCARD_MATCH", "0.40"))
JACCARD_FLOOR = float(os.environ.get("DEDUP_JACCARD_FLOOR", "0.10"))

Classification = Literal["match", "no_match", "borderline"]


def classify(cosine: float, entity_jaccard: float) -> Classification:
    if cosine >= COSINE_MATCH and entity_jaccard >= JACCARD_MATCH:
        return "match"
    if cosine < COSINE_FLOOR or entity_jaccard < JACCARD_FLOOR:
        return "no_match"
    return "borderline"


def decide_pair(
    cosine: float,
    entity_jaccard: float,
    a_title: str,
    a_summary: str,
    a_date: str | None,
    b_title: str,
    b_summary: str,
    b_date: str | None,
) -> bool:
    """Full merge decision for a single candidate pair: threshold rule, falling
    back to LLM adjudication only for borderline cases.
    """
    label = classify(cosine, entity_jaccard)
    if label == "match":
        return True
    if label == "no_match":
        return False

    try:
        result = adjudicate_same_event(a_title, a_summary, a_date, b_title, b_summary, b_date)
        return result.same_event
    except (ExtractionError, APIError) as e:
        # Fail-safe: an incorrect merge corrupts the graph and is hard to
        # undo; a missed merge just leaves a duplicate for a later run to
        # catch. Prefer the recoverable failure mode.
        logger.warning("dedup adjudication failed, treating as no_match: %s", e)
        return False


def find_merge_target(
    candidates: list[dict],
    new_entity_ids: set[str],
    new_title: str,
    new_summary: str,
    new_date: str | None,
) -> str | None:
    """Given candidates sorted by vector score (descending), return the id of
    the first one judged to be the same event, or None if none match.
    """
    for candidate in candidates:
        entity_jaccard = jaccard(new_entity_ids, set(candidate["entity_ids"]))
        is_same = decide_pair(
            cosine=candidate["score"],
            entity_jaccard=entity_jaccard,
            a_title=new_title,
            a_summary=new_summary,
            a_date=new_date,
            b_title=candidate["title"],
            b_summary=candidate["summary"],
            b_date=candidate["date"],
        )
        if is_same:
            return candidate["id"]
    return None
