"""Vector + time-window candidate matching for event deduplication."""

from neo4j import Session

DEFAULT_K = 5
DEFAULT_WINDOW_DAYS = 3
OVERFETCH_K = 20  # ANN query casts a wider net before date-window filtering

CANDIDATE_QUERY = """
CALL db.index.vector.queryNodes('event_embeddings', $overfetch_k, $embedding)
YIELD node AS candidate, score
WHERE candidate.id <> $exclude_id
  AND candidate.date IS NOT NULL AND $date IS NOT NULL
  AND abs(duration.between(date(candidate.date), date($date)).days) <= $window_days
WITH candidate, score
ORDER BY score DESC
LIMIT $k
OPTIONAL MATCH (ent:Entity)-[:PARTICIPATED_IN]->(candidate)
RETURN candidate.id AS id,
       candidate.title AS title,
       candidate.summary AS summary,
       candidate.date AS date,
       score,
       collect(DISTINCT ent.id) AS entity_ids
"""


def find_candidates(
    session: Session,
    embedding: list[float],
    date: str | None,
    exclude_id: str,
    k: int = DEFAULT_K,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[dict]:
    """Find candidate existing events for dedup matching against a new event.

    Returns [] if the new event has no date -- without a date we can't apply
    the time-window filter, and matching purely on embedding similarity across
    the whole graph risks merging unrelated events that happen to read alike.
    """
    if date is None:
        return []
    result = session.run(
        CANDIDATE_QUERY,
        embedding=embedding,
        date=date,
        exclude_id=exclude_id,
        k=k,
        window_days=window_days,
        overfetch_k=OVERFETCH_K,
    )
    return [dict(record) for record in result]
