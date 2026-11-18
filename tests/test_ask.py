"""/ask and /extract with a fake Claude gateway (no network, no cost)."""

import json
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.generation.answer import NO_ANSWER_TEXT, REFUSED_TEXT, AnswerService
from app.generation.extract import ExtractionService
from app.generation.gateway import CitationDelta, Completed, JsonResult, TextDelta
from app.generation.pricing import Usage
from app.generation.prompts import ANSWER_SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT
from tests.helpers import bearer, create_group, create_user, ingest

HR_DOC = b"""# HR Handbook

## Parental leave

Parents receive 16 weeks of paid parental leave.

Secondary caregivers receive 6 weeks.

## Expenses

Hotel stays are capped at 150 EUR per night.
"""

BOARD_DOC = b"""# Board minutes

## Acquisition

The board approved the acquisition of Contoso. Parental leave was not discussed.
"""

QUESTION = "How much parental leave do parents get?"


class FakeGateway:
    def __init__(self, events=(), json_result: JsonResult | None = None) -> None:
        self.events = list(events)
        self.json_result = json_result
        self.calls: list[dict] = []

    async def stream(self, *, system, messages):
        self.calls.append({"system": system, "messages": messages})
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event

    async def create_json(self, *, system, messages, schema):
        self.calls.append({"system": system, "messages": messages, "schema": schema})
        return self.json_result


ANSWER_EVENTS = [
    TextDelta(1, "Parents get 16 weeks "),
    TextDelta(1, "of paid leave"),
    CitationDelta(1, 0, 0, 1, "Parents receive 16 weeks of paid parental leave."),
    TextDelta(2, "."),
    Completed(model="claude-opus-5-5", stop_reason="end_turn", usage=Usage(1000, 50)),
]


@pytest.fixture
def docs(client: TestClient) -> dict:
    hr = create_group("hr")
    return {
        "hr_doc": ingest(client, "hr.md", HR_DOC, group_ids=[hr]),
        "board_doc": ingest(client, "board.md", BOARD_DOC),  # admins only
        "hr_user": create_user("hr@example.com", [hr]),
        "outsider": create_user("new@example.com"),
    }


@pytest.fixture
def install(client: TestClient) -> Iterator:
    """Install answer/extraction services backed by a given fake gateway."""
    state = client.app.state
    original = (state.answer_service, state.extraction_service)

    def _install(gateway: FakeGateway | None) -> FakeGateway | None:
        if gateway is None:  # simulate LLM_PROVIDER=anthropic without a key
            state.answer_service = state.extraction_service = None
            return None
        args = (state.retriever, gateway, state.session_factory, get_settings())
        state.answer_service = AnswerService(*args)
        state.extraction_service = ExtractionService(*args)
        return gateway

    yield _install
    state.answer_service, state.extraction_service = original
    state.retriever.reranker = None


def ask_json(client: TestClient, question: str = QUESTION, user=None, **body) -> dict:
    headers = bearer(user) if user else {}
    response = client.post(
        "/ask", json={"question": question, "stream": False, **body}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    events = []
    for frame in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def blocks_sent(gateway: FakeGateway) -> list[dict]:
    return gateway.calls[0]["messages"][0]["content"]


class TestAsk:
    def test_answer_with_citations(self, client: TestClient, docs, install) -> None:
        install(FakeGateway(ANSWER_EVENTS))
        body = ask_json(client)

        assert body["status"] == "answered"
        assert body["answer"] == "Parents get 16 weeks of paid leave[1]."
        assert body["grounded"] is True
        assert body["segments"][0]["citations"] == [
            {"source": 1, "cited_text": "Parents receive 16 weeks of paid parental leave."}
        ]
        assert body["model"] == "claude-opus-5-5"
        assert body["usage"]["input_tokens"] == 1000
        assert body["cost_usd"] == pytest.approx((1000 * 4 + 50 * 20) / 1_000_000)
        assert "llm_first_token" in body["timings_ms"]
        assert "retrieval_total" in body["timings_ms"]
        source_1 = body["sources"][0]
        assert source_1["number"] == 1 and source_1["label"].startswith("[1] ")

    def test_request_sent_to_claude(self, client: TestClient, docs, install) -> None:
        gateway = install(FakeGateway(ANSWER_EVENTS))
        ask_json(client)

        call = gateway.calls[0]
        assert call["system"] == ANSWER_SYSTEM_PROMPT
        content = blocks_sent(gateway)
        assert content[-1] == {"type": "text", "text": QUESTION}  # question after passages
        results = content[:-1]
        assert results and all(b["type"] == "search_result" for b in results)
        assert all(b["citations"] == {"enabled": True} for b in results)
        assert results[0]["title"].startswith("[1] ")
        # Paragraphs are separate text blocks, so citations can point at one paragraph.
        parental = next(b for b in results if "Parental leave" in b["title"])
        assert len(parental["content"]) == 2

    def test_only_readable_passages_reach_the_model(
        self, client: TestClient, docs, install
    ) -> None:
        gateway = install(FakeGateway(ANSWER_EVENTS))
        body = ask_json(client, user=docs["hr_user"])

        sent_text = json.dumps(blocks_sent(gateway))
        assert "Contoso" not in sent_text
        assert {s["document_id"] for s in body["sources"]} == {docs["hr_doc"]}

    def test_streaming_events(self, client: TestClient, docs, install) -> None:
        install(FakeGateway(ANSWER_EVENTS))
        response = client.post("/ask", json={"question": QUESTION})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(response.text)
        names = [name for name, _ in events]
        assert names[0] == "sources" and names[-1] == "done"
        assert names[1:-1] == ["text", "text", "citation", "text"]
        assert events[3][1] == {
            "block": 1,
            "source": 1,
            "cited_text": "Parents receive 16 weeks of paid parental leave.",
        }
        assert events[-1][1]["answer"] == "Parents get 16 weeks of paid leave[1]."

    def test_no_readable_documents_skips_the_model(self, client: TestClient, docs, install) -> None:
        gateway = install(FakeGateway(ANSWER_EVENTS))
        body = ask_json(client, user=docs["outsider"])
        assert body["status"] == "no_relevant_sources"
        assert body["answer"] == NO_ANSWER_TEXT
        assert body["sources"] == []
        assert body["cost_usd"] == 0.0
        assert gateway.calls == []

    def test_weak_matches_skip_the_model(self, client: TestClient, docs, install) -> None:
        class Irrelevant:
            model_name = "fake"

            def score(self, query, passages):
                return [0.001] * len(passages)

        gateway = install(FakeGateway(ANSWER_EVENTS))
        client.app.state.retriever.reranker = Irrelevant()
        body = ask_json(client, "What is the airspeed of an unladen swallow?")
        assert body["status"] == "no_relevant_sources"
        assert body["sources"]  # passages were found, just not relevant enough
        assert gateway.calls == []

    def test_refusal(self, client: TestClient, docs, install) -> None:
        install(
            FakeGateway(
                [Completed("claude-opus-5-5", "refusal", Usage(800, 0), refusal_category="cyber")]
            )
        )
        body = ask_json(client)
        assert body["status"] == "refused"
        assert body["answer"] == REFUSED_TEXT
        assert body["refusal_category"] == "cyber"
        assert body["grounded"] is False

    def test_truncated_answer_is_flagged(self, client: TestClient, docs, install) -> None:
        install(FakeGateway([TextDelta(1, "Parents get"), Completed("m", "max_tokens", Usage())]))
        assert ask_json(client)["status"] == "truncated"

    def test_failure_mid_stream_sends_error_event(self, client: TestClient, docs, install) -> None:
        install(FakeGateway([TextDelta(1, "Parents"), RuntimeError("connection reset")]))
        events = parse_sse(client.post("/ask", json={"question": QUESTION}).text)
        assert [name for name, _ in events][-1] == "error"
        assert events[-1][1]["status"] == 500
        assert "connection reset" not in json.dumps(events[-1][1])  # no internals leaked

    def test_failure_in_json_mode_returns_error_status(
        self, client: TestClient, docs, install
    ) -> None:
        install(FakeGateway([RuntimeError("boom")]))
        response = client.post("/ask", json={"question": QUESTION, "stream": False})
        assert response.status_code == 500

    def test_unconfigured_returns_503(self, client: TestClient, install) -> None:
        install(None)
        response = client.post("/ask", json={"question": QUESTION})
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

    @pytest.mark.parametrize("payload", [{"question": ""}, {"question": "  "}, {}])
    def test_invalid_requests(self, client: TestClient, payload: dict) -> None:
        assert client.post("/ask", json=payload).status_code == 422


class TestExtract:
    FIELDS = [
        {"name": "hotel_cap", "description": "Max hotel cost per night", "type": "number"},
        {"name": "currency", "description": "Currency of the cap"},
        {"name": "approver", "description": "Who approves exceptions"},
    ]

    def test_extracts_fields_with_sources(self, client: TestClient, docs, install) -> None:
        gateway = install(
            FakeGateway(
                json_result=JsonResult(
                    data={
                        "hotel_cap": {"value": 150, "sources": [1, 99]},
                        "currency": {"value": "EUR", "sources": [1, 1]},
                        "approver": {"value": None, "sources": []},
                    },
                    model="claude-opus-5-5",
                    stop_reason="end_turn",
                    usage=Usage(900, 40),
                )
            )
        )
        response = client.post(
            "/extract", json={"query": "hotel expense policy", "fields": self.FIELDS}
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["status"] == "extracted"
        assert body["fields"]["hotel_cap"] == {"value": 150, "sources": [1]}  # 99 dropped
        assert body["fields"]["currency"] == {"value": "EUR", "sources": [1]}  # deduped
        assert body["fields"]["approver"] == {"value": None, "sources": []}
        assert body["cost_usd"] == pytest.approx((900 * 4 + 40 * 20) / 1_000_000)

        call = gateway.calls[0]
        assert call["system"] == EXTRACT_SYSTEM_PROMPT
        schema = call["schema"]
        assert schema["required"] == ["hotel_cap", "currency", "approver"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["hotel_cap"]["properties"]["value"] == {
            "anyOf": [{"type": "number"}, {"type": "null"}]
        }
        # Citations can't be combined with structured outputs.
        results = blocks_sent(gateway)[:-1]
        assert all(b["citations"] == {"enabled": False} for b in results)
        assert "hotel_cap (number): Max hotel cost per night" in blocks_sent(gateway)[-1]["text"]

    def test_no_readable_documents(self, client: TestClient, docs, install) -> None:
        gateway = install(FakeGateway())
        response = client.post(
            "/extract",
            json={"query": "hotel cap", "fields": self.FIELDS[:1]},
            headers=bearer(docs["outsider"]),
        )
        body = response.json()
        assert body["status"] == "no_relevant_sources"
        assert body["fields"] == {"hotel_cap": {"value": None, "sources": []}}
        assert gateway.calls == []

    @pytest.mark.parametrize(
        "fields",
        [
            [{"name": "Bad Name", "description": "x"}],
            [{"name": "a", "description": "x"}, {"name": "a", "description": "y"}],
            [{"name": "a", "description": "x", "type": "date"}],
            [],
        ],
    )
    def test_invalid_fields(self, client: TestClient, fields: list) -> None:
        response = client.post("/extract", json={"query": "q", "fields": fields})
        assert response.status_code == 422

    def test_unconfigured_returns_503(self, client: TestClient, install) -> None:
        install(None)
        response = client.post("/extract", json={"query": "q", "fields": self.FIELDS[:1]})
        assert response.status_code == 503


def test_random_document_filter_is_harmless(client: TestClient, docs, install) -> None:
    gateway = install(FakeGateway(ANSWER_EVENTS))
    body = ask_json(client, document_ids=[str(uuid.uuid4())])
    assert body["status"] == "no_relevant_sources"
    assert gateway.calls == []
