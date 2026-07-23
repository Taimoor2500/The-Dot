"""Bootstrap Neo4j constraints and the event vector index. Idempotent."""

from extraction.llm import EMBEDDING_DIM
from graph.client import session

STATEMENTS = [
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE",
    f"""
    CREATE VECTOR INDEX event_embeddings IF NOT EXISTS
    FOR (e:Event) ON e.embedding
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIM},
        `vector.similarity_function`: 'cosine'
    }}}}
    """,
]


def bootstrap() -> None:
    with session() as s:
        for statement in STATEMENTS:
            s.run(statement)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    bootstrap()
    print("Neo4j schema bootstrapped (constraints + vector index).")
