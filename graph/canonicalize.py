"""Entity canonicalization v0: static alias map + normalization.

LLM-assisted canonicalization (fuzzy duplicate merging) comes in Phase 3.
This is intentionally simple: a fixed alias table for well-known countries/orgs
plus uppercase/whitespace normalization for everything else.
"""

ALIASES: dict[str, str] = {
    "u.s.": "United States",
    "u.s.a.": "United States",
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "eu": "European Union",
    "e.u.": "European Union",
    "prc": "China",
    "people's republic of china": "China",
    "un": "United Nations",
    "u.n.": "United Nations",
    "nato": "NATO",
}


def canonicalize_name(name: str) -> str:
    """Return the canonical form of an entity name."""
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    return name.strip()


def canonical_id(entity_type: str, name: str) -> str:
    """Stable id for an entity node, used as the merge key on write."""
    canonical_name = canonicalize_name(name)
    return f"{entity_type}:{canonical_name.lower()}"
