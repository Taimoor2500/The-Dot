"""Score the dedup merge-decision pipeline (threshold rule + LLM adjudication)
against evals/dedup_gold_set.jsonl -- hand-labeled same-event/different-event
pairs.

Usage:
    uv run python -m evals.score_dedup

Note: each pair costs 2 embedding calls plus, for borderline pairs, 1
adjudication call -- cheap, but not free. This is "gold set #1" from the
plan; keep adding real labeled pairs (from articles that turn out to cover
the same event) as daily ingestion accumulates duplicate coverage.
"""

import json
import math
from pathlib import Path

from dotenv import load_dotenv

from dedup.decide import decide_pair
from dedup.jaccard import jaccard
from extraction.llm import embed_text
from graph.canonicalize import canonical_id

GOLD_SET_PATH = Path(__file__).parent / "dedup_gold_set.jsonl"


def _entity_ids(entities: list[dict]) -> set[str]:
    return {canonical_id(e["type"], e["name"]) for e in entities}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score(gold_set_path: Path = GOLD_SET_PATH) -> None:
    load_dotenv()
    tp = fp = fn = tn = 0

    with open(gold_set_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    for i, pair in enumerate(pairs, start=1):
        a, b, label = pair["event_a"], pair["event_b"], pair["label"]

        emb_a = embed_text(f"{a['title']}\n{a['summary']}")
        emb_b = embed_text(f"{b['title']}\n{b['summary']}")
        cosine = _cosine(emb_a, emb_b)
        entity_jaccard = jaccard(_entity_ids(a["entities"]), _entity_ids(b["entities"]))

        predicted = decide_pair(
            cosine=cosine,
            entity_jaccard=entity_jaccard,
            a_title=a["title"],
            a_summary=a["summary"],
            a_date=a.get("date"),
            b_title=b["title"],
            b_summary=b["summary"],
            b_date=b.get("date"),
        )

        outcome = "correct" if predicted == label else "WRONG"
        print(
            f"[{i}] cosine={cosine:.3f} jaccard={entity_jaccard:.3f} "
            f"predicted={predicted} actual={label} ({outcome})"
        )

        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0
        else math.nan
    )

    print(f"\n{len(pairs)} pairs -- TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    score()
