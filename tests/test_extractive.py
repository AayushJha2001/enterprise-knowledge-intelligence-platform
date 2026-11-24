"""Offline extractive answerer: unit tests and end to end through /ask and /extract."""

import asyncio
import uuid

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.generation.extractive import (
    MODEL_NAME,
    NO_DIRECT_ANSWER,
    ExtractiveGateway,
    candidate_sentences,
)
from app.generation.gateway import (
    AnthropicGateway,
    CitationDelta,
    Completed,
    TextDelta,
    build_gateway,
)
from app.generation.prompts import Source, user_message
from app.ingestion.embeddings import HashingEmbedder
from tests.helpers import ingest

EMBEDDER = HashingEmbedder(384)


def source(number: int, content: str, title: str = "Handbook") -> Source:
    return Source(number, uuid.uuid4(), uuid.uuid4(), title, None, None, None, 1.0, content)


def run_stream(gateway: ExtractiveGateway, sources: list[Source], question: str) -> list:
    async def go():
        messages = [user_message(sources, question, citations=True)]
        return [e async for e in gateway.stream(system="", messages=messages)]

    return asyncio.run(go())


SOURCES = [
    source(
        1,
        "Employees may work remotely up to three days per week.\n\n"
        "Parents receive sixteen weeks of paid parental leave. "
        "Leave must be requested eight weeks in advance.",
    ),
    source(2, "Hotel stays are capped at 150 EUR per night in most cities."),
]


def test_picks_the_matching_sentence_and_cites_its_paragraph() -> None:
    gateway = ExtractiveGateway(EMBEDDER, max_sentences=1)
    events = run_stream(
        gateway, SOURCES, "how many weeks of paid parental leave do parents receive"
    )
    text = [e for e in events if isinstance(e, TextDelta)]
    citations = [e for e in events if isinstance(e, CitationDelta)]

    assert [t.text for t in text] == ["Parents receive sixteen weeks of paid parental leave."]
    assert citations == [
        CitationDelta(
            block_index=0,
            search_result_index=0,
            start_block_index=1,  # second paragraph of the first passage
            end_block_index=2,
            cited_text=(
                "Parents receive sixteen weeks of paid parental leave. "
                "Leave must be requested eight weeks in advance."
            ),
        )
    ]
    assert events[-1] == Completed(model=MODEL_NAME, stop_reason="end_turn", usage=events[-1].usage)


def test_several_sentences_keep_document_order() -> None:
    gateway = ExtractiveGateway(EMBEDDER, max_sentences=3, margin=10.0)  # take top 3
    events = run_stream(gateway, SOURCES, "parental leave weeks hotel")
    sentences = [e.text.strip() for e in events if isinstance(e, TextDelta)]
    assert len(sentences) == 3
    order = [c.search_result_index for c in events if isinstance(c, CitationDelta)]
    assert order == sorted(order)


def test_no_usable_sentences() -> None:
    events = run_stream(ExtractiveGateway(EMBEDDER), [source(1, "Too short.")], "anything")
    assert events[0] == TextDelta(0, NO_DIRECT_ANSWER)
    assert not any(isinstance(e, CitationDelta) for e in events)


def test_wrapped_lines_are_joined_but_list_items_kept() -> None:
    passages = [
        [
            "Hotel stays are capped at\n150 EUR per night, or 220 EUR\nin London and Paris.",
            "- First item is here now\n- Second item is here now",
        ]
    ]
    assert [c.sentence for c in candidate_sentences(passages)] == [
        "Hotel stays are capped at 150 EUR per night, or 220 EUR in London and Paris.",
        "First item is here now",
        "Second item is here now",
    ]


def test_answer_markdown_puts_markers_after_each_sentence(client: TestClient) -> None:
    ingest(client, "hr.md", HANDBOOK)
    gateway = client.app.state.answer_service.gateway
    gateway.margin, gateway.max_sentences = 10.0, 2  # force two sentences
    try:
        body = client.post(
            "/ask", json={"question": "parental leave hotel", "stream": False}
        ).json()
    finally:
        gateway.margin, gateway.max_sentences = 0.06, 3
    assert body["answer"].count("[") == 2
    assert ".[" in body["answer"] and "] " in body["answer"]  # "First.[1] Second.[1]"


def test_source_label_does_not_repeat_the_title() -> None:
    from app.retrieval.search import location

    assert location("Handbook", "Handbook > Leave > Annual") == "Handbook > Leave > Annual"
    assert location("Handbook", "Leave") == "Handbook > Leave"
    assert location("Handbook", None) == "Handbook"
    assert location(None, "Leave > Annual") == "Leave > Annual"


def test_candidates_skip_code_fragments_and_duplicates() -> None:
    passages = [
        ["```python\nprint('hello there friend')\n```", "Same sentence appears twice here."],
        ["Same sentence appears twice here.", "Tiny bit."],
    ]
    assert [c.sentence for c in candidate_sentences(passages)] == [
        "Same sentence appears twice here."
    ]


def test_extract_fills_fields_from_best_sentences() -> None:
    gateway = ExtractiveGateway(EMBEDDER)
    schema = {
        "properties": {
            "hotel_cap": {
                "description": "hotel stays capped EUR per night",
                "properties": {"value": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
            },
            "remote": {
                "description": "allowed to work remotely",
                "properties": {"value": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}},
            },
        }
    }

    async def go():
        messages = [user_message(SOURCES, "Topic: policies\n\nFields...", citations=False)]
        return await gateway.create_json(system="", messages=messages, schema=schema)

    result = asyncio.run(go())
    assert result.data["hotel_cap"] == {"value": 150.0, "sources": [2]}
    assert result.data["remote"] == {"value": None, "sources": []}  # booleans not inferred
    assert result.model == MODEL_NAME


def test_provider_selection() -> None:
    assert isinstance(build_gateway(Settings(anthropic_api_key=""), EMBEDDER), ExtractiveGateway)
    with_key = Settings(anthropic_api_key="sk-test")
    assert isinstance(build_gateway(with_key, EMBEDDER), AnthropicGateway)
    forced = Settings(anthropic_api_key="sk-test", llm_provider="extractive")
    assert isinstance(build_gateway(forced, EMBEDDER), ExtractiveGateway)
    assert build_gateway(Settings(llm_provider="anthropic", anthropic_api_key=""), EMBEDDER) is None


HANDBOOK = b"""# HR Handbook

## Parental leave

Parents receive sixteen weeks of paid parental leave. Requests go through the HR portal.

## Expenses

Hotel stays are capped at 150 EUR per night in most cities.
"""


def test_ask_end_to_end_without_an_api(client: TestClient) -> None:
    ingest(client, "hr.md", HANDBOOK)
    response = client.post(
        "/ask", json={"question": "weeks of paid parental leave for parents", "stream": False}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["model"] == MODEL_NAME
    assert body["cost_usd"] == 0.0
    assert body["grounded"] is True
    assert "sixteen weeks of paid parental leave" in body["answer"]
    cited = body["segments"][0]["citations"][0]["source"]
    assert "Parental leave" in body["sources"][cited - 1]["label"]


def test_extract_end_to_end_without_an_api(client: TestClient) -> None:
    ingest(client, "hr.md", HANDBOOK)
    response = client.post(
        "/extract",
        json={
            "query": "hotel expense policy",
            "fields": [
                {"name": "cap", "description": "hotel stays capped per night", "type": "number"}
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["fields"]["cap"]["value"] == 150.0
