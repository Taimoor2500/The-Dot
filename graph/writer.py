"""Batched UNWIND writer: ExtractedEvent -> Event/Entity/Claim/Source graph."""

import hashlib

from neo4j import Session

from extraction.schemas import ExtractedEvent
from graph.canonicalize import canonical_id, canonicalize_name

WRITE_BATCH_QUERY = """
UNWIND $events AS ev
MERGE (e:Event {id: ev.id})
SET e.title = ev.title,
    e.summary = ev.summary,
    e.date = ev.date,
    e.location = ev.location,
    e.confidence = ev.confidence,
    e.embedding = ev.embedding,
    e.corroboration_count = coalesce(e.corroboration_count, 1)
MERGE (src:Source {url: ev.source_url})
SET src.domain = ev.domain
MERGE (e)-[:REPORTED_BY]->(src)
WITH e, ev
UNWIND ev.entities AS ent
MERGE (n:Entity {id: ent.id})
SET n.name = ent.name, n.type = ent.type
MERGE (n)-[r:PARTICIPATED_IN]->(e)
SET r.role = ent.role
WITH e, ev
UNWIND ev.claims AS cl
MERGE (c:Claim {id: cl.id})
SET c.text = cl.text, c.source_url = cl.source_url
MERGE (c)-[:SUPPORTS]->(e)
"""


def _event_id(source_url: str) -> str:
    return "event:" + hashlib.sha256(source_url.encode()).hexdigest()[:16]


def _claim_id(source_url: str, text: str) -> str:
    return "claim:" + hashlib.sha256(f"{source_url}|{text}".encode()).hexdigest()[:16]


def build_payload(
    event: ExtractedEvent,
    embedding: list[float],
    source_url: str,
    domain: str | None,
) -> dict:
    """Convert a validated ExtractedEvent into the flat dict shape the writer expects."""
    return {
        "id": _event_id(source_url),
        "title": event.title,
        "summary": event.summary,
        "date": event.date.isoformat() if event.date else None,
        "location": event.location,
        "confidence": event.confidence,
        "embedding": embedding,
        "source_url": source_url,
        "domain": domain,
        "entities": [
            {
                "id": canonical_id(ent.type, ent.name),
                "name": canonicalize_name(ent.name),
                "type": ent.type,
                "role": ent.role,
            }
            for ent in event.entities
        ],
        "claims": [
            {
                "id": _claim_id(source_url, claim.text),
                "text": claim.text,
                "source_url": claim.source_url,
            }
            for claim in event.claims
        ],
    }


def write_events_batch(neo4j_session: Session, payloads: list[dict]) -> None:
    if not payloads:
        return
    neo4j_session.run(WRITE_BATCH_QUERY, events=payloads)
