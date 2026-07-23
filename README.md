# The Dot

An AI-powered app that ingests world news, extracts structured events, and reveals how
events connect — causes, responses, storylines — through a sourced knowledge graph.

Domain scope for v1: **geopolitics & trade**, last ~60 days.

## Phase 1 — Skeleton & First Graph

Pipeline: GDELT DOC API (article metadata) → fetch article text → Gemini extraction
(schema-validated JSON) → Neo4j graph (Events, Entities, Claims, Sources).

### Setup

1. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):
   ```
   uv sync
   ```

2. **Get a Gemini API key**: https://aistudio.google.com/apikey (free tier, no card required).

3. **Create a free Neo4j AuraDB instance**:
   - Go to https://console.neo4j.io and sign up / log in.
   - Click "New Instance" → choose **AuraDB Free**.
   - Name it (e.g. `the-dot-dev`) and create it.
   - When the instance is created, download/copy the generated credentials
     (connection URI, username `neo4j`, and password) — Aura only shows the
     password once.
   - Wait for the instance status to show "Running" (~1-2 minutes).

4. **Configure environment**:
   ```
   cp .env.example .env
   ```
   Fill in `GEMINI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.

5. **Bootstrap the Neo4j schema** (constraints + vector index):
   ```
   uv run python -m graph.schema
   ```

6. **Run the pipeline**:
   ```
   uv run python -m ingestion.run --limit 100
   ```

7. **Inspect the graph**: open your Aura instance's "Query" tab (or Neo4j Browser) and run:
   ```cypher
   MATCH (e:Event)-[:REPORTED_BY]->(s:Source)
   RETURN e, s LIMIT 50
   ```

### Run tests

```
uv run pytest
```

Tests cover pure logic (entity canonicalization, payload construction, schema
validation) with mocked/no network calls. The full pipeline is verified by running
it manually against a small batch, since each run costs real Gemini API calls.

### Repo layout

```
ingestion/    # GDELT fetch, article text extraction, SQLite article buffer, CLI entrypoint
extraction/   # Pydantic schemas, Gemini client (extraction + embeddings), retry pipeline
graph/        # Neo4j client, schema bootstrap, entity canonicalization, batched writer
tests/        # pytest unit tests
```

Later phases add `dedup/`, `connections/`, `retrieval/`, `api/`, `web/`, `evals/`.
