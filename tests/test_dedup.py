from dedup.decide import classify, find_merge_target
from dedup.jaccard import jaccard


def test_jaccard_identical_sets():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial_overlap():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_jaccard_both_empty_is_zero_not_undefined():
    assert jaccard(set(), set()) == 0.0


def test_classify_clear_match():
    assert classify(cosine=0.95, entity_jaccard=0.8) == "match"


def test_classify_clear_non_match_on_low_cosine():
    assert classify(cosine=0.3, entity_jaccard=0.9) == "no_match"


def test_classify_clear_non_match_on_low_jaccard():
    assert classify(cosine=0.95, entity_jaccard=0.0) == "no_match"


def test_classify_borderline():
    assert classify(cosine=0.80, entity_jaccard=0.25) == "borderline"


def test_find_merge_target_returns_first_clear_match_without_calling_llm(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("adjudicate_same_event should not be called for a clear match")

    monkeypatch.setattr("dedup.decide.adjudicate_same_event", fail_if_called)

    candidates = [
        {"id": "event:1", "title": "A", "summary": "s", "date": "2026-01-01", "score": 0.95, "entity_ids": ["x", "y"]},
    ]
    result = find_merge_target(candidates, {"x", "y"}, "A2", "s2", "2026-01-01")
    assert result == "event:1"


def test_find_merge_target_returns_none_when_no_candidates_match():
    candidates = [
        {"id": "event:1", "title": "A", "summary": "s", "date": "2026-01-01", "score": 0.1, "entity_ids": ["z"]},
    ]
    result = find_merge_target(candidates, {"x", "y"}, "A2", "s2", "2026-01-01")
    assert result is None
