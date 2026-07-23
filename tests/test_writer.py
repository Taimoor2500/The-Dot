from datetime import date

from extraction.schemas import Claim, Entity, ExtractedEvent
from graph.writer import _event_id, build_payload


def make_event() -> ExtractedEvent:
    return ExtractedEvent(
        title="Country A sanctions Country B",
        summary="Country A imposed new trade sanctions on Country B.",
        date=date(2026, 7, 1),
        location="Country A",
        entities=[Entity(name="U.S.", type="COUNTRY", role="actor")],
        claims=[Claim(text="Sanctions took effect immediately.", source_url="https://example.com/a")],
        confidence=0.9,
    )


def test_build_payload_shapes_event_for_writer():
    event = make_event()
    embedding = [0.1, 0.2, 0.3]
    payload = build_payload(event, embedding, "https://example.com/a", "example.com")

    assert payload["id"] == _event_id("https://example.com/a")
    assert payload["title"] == event.title
    assert payload["date"] == "2026-07-01"
    assert payload["embedding"] == embedding
    assert payload["entities"][0]["name"] == "United States"
    assert payload["entities"][0]["id"].startswith("COUNTRY:")
    assert len(payload["claims"]) == 1
    assert payload["claims"][0]["id"].startswith("claim:")


def test_event_id_is_stable_for_same_url():
    assert _event_id("https://example.com/a") == _event_id("https://example.com/a")
    assert _event_id("https://example.com/a") != _event_id("https://example.com/b")
