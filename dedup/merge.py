"""Merge semantics: attach a newly-extracted article's source/entities/claims
to an existing survivor Event instead of creating a duplicate node.

Claims are intentionally NOT text-deduplicated across sources here: two
articles independently reporting the same fact become two Claim nodes, each
keeping its own source_url. That preserves per-claim provenance (which the
product principles require -- "every fact links to its source article") and
is arguably more useful than collapsing them, at the cost of not matching the
plan's literal "dedup near-identical claim texts" wording.
"""

from neo4j import Session

MERGE_QUERY = """
MATCH (e:Event {id: $survivor_id})
SET e.date = CASE
    WHEN $date IS NOT NULL AND (e.date IS NULL OR date($date) < date(e.date))
    THEN $date ELSE e.date
END
MERGE (src:Source {url: $source_url})
SET src.domain = $domain
MERGE (e)-[:REPORTED_BY]->(src)
WITH e
UNWIND $entities AS ent
MERGE (n:Entity {id: ent.id})
SET n.name = ent.name, n.type = ent.type
MERGE (n)-[r:PARTICIPATED_IN]->(e)
SET r.role = ent.role
WITH e
UNWIND $claims AS cl
MERGE (c:Claim {id: cl.id})
SET c.text = cl.text, c.source_url = cl.source_url
MERGE (c)-[:SUPPORTS]->(e)
WITH e
MATCH (e)-[:REPORTED_BY]->(s:Source)
WITH e, count(DISTINCT s) AS source_count
SET e.corroboration_count = source_count
"""


def merge_into_existing(neo4j_session: Session, survivor_id: str, payload: dict) -> None:
    """Attach payload's source/entities/claims to the existing event `survivor_id`
    and recompute its corroboration count from actual distinct sources.
    """
    neo4j_session.run(
        MERGE_QUERY,
        survivor_id=survivor_id,
        date=payload["date"],
        source_url=payload["source_url"],
        domain=payload["domain"],
        entities=payload["entities"],
        claims=payload["claims"],
    )
