from graph.canonicalize import canonical_id, canonicalize_name


def test_known_alias_normalizes_to_canonical_name():
    assert canonicalize_name("U.S.") == "United States"
    assert canonicalize_name("usa") == "United States"
    assert canonicalize_name("United States") == "United States"


def test_unknown_name_is_stripped_but_unchanged():
    assert canonicalize_name("  Acme Corp  ") == "Acme Corp"


def test_canonical_id_is_stable_across_aliases():
    assert canonical_id("COUNTRY", "U.S.") == canonical_id("COUNTRY", "USA")
    assert canonical_id("COUNTRY", "U.S.") != canonical_id("ORG", "U.S.")
