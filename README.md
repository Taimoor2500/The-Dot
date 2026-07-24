# The Dot

> From "connecting the dots": every news event is a dot; this project reveals the lines between them.

An AI-powered pipeline that ingests world news, extracts structured events into a
knowledge graph, and (in later phases) answers multi-hop questions about how events
connect  causes, responses, storylines — with full source provenance on every fact.

**Domain scope for v1:** geopolitics & trade, rolling ~60-day window. Narrow on purpose:
a small, accurate, well-sourced graph beats a large noisy one.

**Status:** Phases 1–2 of 6 complete (ingestion/extraction, deduplication). See [Roadmap](#roadmap).

---

## The problem

News is consumed as isolated headlines. The connections between events — a sanctions
decision tracing back to a shipping incident, a diplomatic visit responding to a trade
ban — live only in analysts' heads. Standard vector RAG can't answer "how is event A
connected to event B?" because similarity search misses causal chains. This project
builds an event-centric knowledge graph instead: every event is a node, every
connection is a typed edge, and every fact carries provenance back to its source.

## Architecture (current)

```
GDELT DOC 2.0 API (geopolitics & trade query, 60-day window)
        │  metadata only: url, title, domain, date
        ▼
ingestion/fetch_text.py   trafilatura: HTML → clean article text (concurrent, 10-way)
        │
        ▼
ingestion/store.py        SQLite article buffer (idempotent by URL)
        │
        ▼
extraction/llm.py         Gemini (gemini-2.5-flash), schema-constrained JSON output
        │                 → title, summary, date, location, entities[], claims[], confidence
        │                 + gemini-embedding-001 embeds (title+summary), 768-dim
        ▼
dedup/matching.py         vector top-k search + ±3 day window against existing graph
        │
        ▼
dedup/decide.py           cosine + entity-Jaccard threshold rule;
        │                 borderline cases → cheap LLM adjudication call
        │
   ┌────┴────┐
   ▼         ▼
dedup/     graph/writer.py
merge.py   (new event)
   │         │
   └────┬────┘
        ▼
Neo4j AuraDB (Event / Entity / Claim / Source graph + vector index)
```

One CLI command runs the whole thing: `uv run python -m ingestion.run --limit 100`.

## Data model

| Node | Created from | Key properties |
|---|---|---|
| `Event` | One per real-world event (after dedup) | `title`, `summary`, `date`, `location`, `confidence`, `embedding`, `corroboration_count` |
| `Entity` | Extracted per event, canonicalized across events | `name`, `type` (`PERSON`/`ORG`/`COUNTRY`/`COMPANY`) |
| `Claim` | Atomic facts extracted per event | `text`, `source_url` |
| `Source` | The originating article | `url`, `domain` |

| Edge | Direction | Meaning |
|---|---|---|
| `PARTICIPATED_IN` | `Entity → Event` | has a `role` property (actor/target/mediator) |
| `REPORTED_BY` | `Event → Source` | provenance; an event can have multiple sources after dedup |
| `SUPPORTS` | `Claim → Event` | which claims back this event |

This is a custom ontology defined for this project (not schema.org/Wikidata/GDELT's
CAMEO coding), enforced at extraction time via Pydantic schemas
(`extraction/schemas.py`) and Gemini's structured-output mode. Event-to-event edges
(`CAUSED`, `RESPONDED_TO`, `PRECEDED`) and storylines are Phase 3 — not built yet.

## Deduplication (Phase 2)

Ten articles covering the same summit should become **one** `Event` node with ten
sources, not ten near-duplicate events. For each newly extracted event:

1. **Candidate matching** (`dedup/matching.py`): vector-search the existing graph for
   similar events within a ±3-day window (not the whole graph — cheaper, and avoids
   matching unrelated events that happen to read alike).
2. **Decision** (`dedup/decide.py`): classify each candidate pair by cosine similarity
   + entity-Jaccard overlap into three bands:
   - **Clear match** (cosine ≥ 0.90 and Jaccard ≥ 0.40) → merge automatically.
   - **Clear non-match** (cosine < 0.70 or Jaccard < 0.10) → skip, no LLM call.
   - **Borderline** → a cheap LLM adjudication call (`extraction/llm.py:adjudicate_same_event`)
     decides. This is the case a hard threshold gets wrong — e.g. two articles about
     related-but-distinct Iran/Houthi incidents share entities and read similarly, but
     aren't the same event.
   - Thresholds are placeholder defaults (`DEDUP_COSINE_MATCH`, `DEDUP_COSINE_FLOOR`,
     `DEDUP_JACCARD_MATCH`, `DEDUP_JACCARD_FLOOR` env vars) pending real tuning against
     a larger gold set.
3. **Merge** (`dedup/merge.py`): attach the new article's source/entities/claims to the
   existing survivor event, keep the earliest date, recompute `corroboration_count`
   from actual distinct sources.
4. **Fail-safe on adjudication errors**: any LLM/API failure during adjudication is
   treated as "not a match." A missed merge just leaves a duplicate for a later run to
   catch; an incorrect merge corrupts the graph and is hard to undo.

**Known simplification:** claims are *not* text-deduplicated across merged sources —
two articles independently reporting the same fact become two `Claim` nodes, each
keeping its own `source_url`, rather than collapsing into one. Chosen deliberately to
preserve per-claim provenance ("every fact links to its source article" is a
non-negotiable product principle) over the plan's literal wording.

**Writes are per-event, not batched**, unlike Phase 1: dedup candidate matching needs
to see events written earlier in the *same* run, so a new event must be visible in
Neo4j before the next article's candidate search runs. At ~100 articles/run the extra
round trips are cheap relative to the Gemini calls that dominate runtime anyway.

Verified working on real data: a wire-service story ("envoys hold talks in Pakistan")
appearing under two different domains merged into one `Event` node with
`corroboration_count = 2`, rather than creating a duplicate.

## Evals

`evals/dedup_gold_set.jsonl` + `evals/score_dedup.py` — hand-labeled same-event/
different-event pairs, scored for precision/recall/F1.

```
uv run python -m evals.score_dedup
```

Current: **4 pairs, precision=1.0, recall=1.0, F1=1.0** (1 synthetic same-event
paraphrase + 3 real different-event pairs, including a deliberate hard negative). This
confirms the mechanism works end-to-end, not that thresholds are well-tuned — the plan
calls for ~100 hand-labeled pairs; this grows as daily ingestion surfaces genuine
duplicate coverage.

## Setup

1. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):
   ```
   uv sync
   ```

2. **Get a Gemini API key**: https://aistudio.google.com/apikey (free tier, no card
   required). Note: the free tier caps `gemini-2.5-flash` at a low daily request quota
   (as low as 20/day on some accounts) — the pipeline handles this gracefully (see
   [Operational notes](#operational-notes)) but don't expect to process hundreds of
   articles per day without enabling billing.

3. **Create a free Neo4j AuraDB instance**:
   - Go to https://console.neo4j.io and sign up / log in.
   - Click "New Instance" → choose **AuraDB Free** (not the Professional trial, which
     the console surfaces more prominently and expires in 14 days).
   - Name it (e.g. `the-dot-dev`) and create it.
   - Copy the generated credentials (connection URI, username `neo4j`, password) —
     Aura only shows the password once.
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

7. **Inspect the graph** in the Aura console's Query tab:
   ```cypher
   MATCH (e:Event)-[r:REPORTED_BY]->(s:Source)
   OPTIONAL MATCH (ent:Entity)-[r2:PARTICIPATED_IN]->(e)
   RETURN e, r, s, ent, r2 LIMIT 100
   ```
   (Include relationship variables in `RETURN` — Neo4j Browser only draws edges it
   sees in the result, not just co-present nodes.)

## Run tests

```
uv run pytest
```

18 tests covering pure logic (entity canonicalization, payload construction, Jaccard,
threshold classification, quota-error detection) with mocked/no network calls. The
full pipeline is verified by running it manually, since each run costs real Gemini API
calls and depends on GDELT/Neo4j availability.

## Operational notes

A few real issues hit during development, worth knowing if you extend this:

- **GDELT rate-limits by IP**, not API key (it's a free, unauthenticated endpoint). A
  shared/cloud IP can get throttled by other traffic entirely unrelated to this
  project; the fetcher retries patiently (`ingestion/gdelt.py`) since this is a batch
  job, not latency-sensitive.
- **Gemini's free tier daily quota is low** (`gemini-2.5-flash`: as low as 20
  requests/day on some accounts). Quota-exhausted articles are left unmarked in
  SQLite (not permanently skipped) so a future run picks them up — see
  `extraction/llm.QuotaExhaustedError` and its handling in `ingestion/run.py`.
- **Gemini API calls have no default timeout** (`google-genai` passes `timeout=None`
  to httpx by default) — a stalled connection hangs forever with no error. Fixed with
  an explicit 60s timeout (`extraction/llm.py:REQUEST_TIMEOUT_MS`).
- **`genai.Client()` must be reused, not constructed per call** — a fresh client gets
  garbage-collected mid-request (its httpx client closes itself in `__del__`).

## Repo layout

```
ingestion/    # GDELT fetch, article text extraction, SQLite article buffer, CLI entrypoint
extraction/   # Pydantic schemas, Gemini client (extraction + embeddings + adjudication)
graph/        # Neo4j client, schema bootstrap, entity canonicalization, batched writer
dedup/        # candidate matching, merge decision (threshold + LLM), merge semantics
evals/        # gold sets + scoring scripts
tests/        # pytest unit tests
```

Later phases add `connections/`, `retrieval/`, `api/`, `web/`.

## Roadmap

- **Phase 3 — Connections**: explicit edges (`PRECEDED`, `RESPONDED_TO`) from
  extraction, inferred edges (`CAUSED`) via LLM judgment with confidence + rationale,
  storyline detection (community detection), entity canonicalization v2 (LLM-assisted
  fuzzy merging).
- **Phase 4 — GraphRAG retrieval**: local search (vector + graph expansion), global
  search (storyline summaries), Connection Explainer (shortest-path narration), a
  naive baseline vector-RAG for comparison.
- **Phase 5 — Frontend**: graph view, timeline view, Q&A box, Connection Explainer UI.
- **Phase 6 — Deployment & writeup**: hosted demo, final eval run, README polish, demo
  video.
