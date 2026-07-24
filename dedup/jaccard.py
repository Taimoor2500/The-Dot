"""Entity-overlap similarity for the dedup merge decision."""


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets of canonical entity ids.

    Two events with no entities at all are not considered similar by this
    measure (0.0, not 1.0/undefined) -- an empty-vs-empty match carries no
    evidence either way.
    """
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
