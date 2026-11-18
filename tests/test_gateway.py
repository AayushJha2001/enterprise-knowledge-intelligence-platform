"""ClaudeGateway: request parameters and translation of SDK stream events."""

import asyncio
from types import SimpleNamespace as NS

from app.core.config import Settings
from app.generation.gateway import (
    FALLBACK_BETA,
    CitationDelta,
    ClaudeGateway,
    Completed,
    TextDelta,
)
from app.generation.pricing import Usage, estimate_cost_usd


class FakeStream:
    def __init__(self, events, final) -> None:
        self.events, self.final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, events, final) -> None:
        self.params: dict = {}
        outer = self

        class Messages:
            def stream(self, **params):
                outer.params = params
                return FakeStream(events, final)

        self.beta = NS(messages=Messages())


def delta(index, **fields):
    return NS(type="content_block_delta", index=index, delta=NS(**fields))


EVENTS = [
    NS(type="message_start"),
    delta(0, type="thinking_delta", thinking=""),
    delta(1, type="text_delta", text="16 weeks"),
    delta(
        1,
        type="citations_delta",
        citation=NS(
            type="search_result_location",
            search_result_index=2,
            start_block_index=0,
            end_block_index=1,
            cited_text="Parents receive 16 weeks.",
        ),
    ),
    delta(1, type="citations_delta", citation=NS(type="char_location")),  # not ours: ignored
    NS(type="message_stop"),
]
FINAL = NS(
    model="claude-opus-5-5",
    stop_reason="end_turn",
    usage=NS(input_tokens=1200, output_tokens=80, cache_read_input_tokens=None),
    _request_id="req_123",
)


def collect(gateway: ClaudeGateway) -> list:
    async def run():
        return [e async for e in gateway.stream(system="sys", messages=[{"role": "user"}])]

    return asyncio.run(run())


def test_translates_stream_events() -> None:
    events = collect(ClaudeGateway(FakeClient(EVENTS, FINAL), Settings()))
    assert events == [
        TextDelta(1, "16 weeks"),
        CitationDelta(1, 2, 0, 1, "Parents receive 16 weeks."),
        Completed(
            model="claude-opus-5-5",
            stop_reason="end_turn",
            usage=Usage(input_tokens=1200, output_tokens=80),
            request_id="req_123",
        ),
    ]


def test_request_parameters() -> None:
    client = FakeClient([], FINAL)
    collect(ClaudeGateway(client, Settings(answer_effort="medium")))
    assert client.params["model"] == "claude-opus-5-5"
    assert client.params["system"] == "sys"
    assert client.params["output_config"] == {"effort": "medium"}
    assert client.params["betas"] == [FALLBACK_BETA]
    assert client.params["fallbacks"] == "default"
    assert "thinking" not in client.params  # Opus 5.5 always uses adaptive thinking


def test_fallbacks_can_be_disabled() -> None:
    client = FakeClient([], FINAL)
    collect(ClaudeGateway(client, Settings(llm_fallbacks_enabled=False)))
    assert "fallbacks" not in client.params and "betas" not in client.params


def test_refusal_category_is_reported() -> None:
    final = NS(
        model="claude-opus-5-5",
        stop_reason="refusal",
        usage=NS(input_tokens=10, output_tokens=0),
        stop_details=NS(category="cyber"),
    )
    events = collect(ClaudeGateway(FakeClient([], final), Settings()))
    assert events[-1].refusal_category == "cyber"


def test_cost_estimate() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=0,
    )
    assert estimate_cost_usd("claude-opus-5-5", usage) == 4.0 + 2.0 + 0.2
    assert estimate_cost_usd("some-unknown-model", usage) is None


def test_empty_api_key_means_unconfigured() -> None:
    assert Settings(anthropic_api_key="").anthropic_api_key is None
    assert Settings(anthropic_api_key="  ").anthropic_api_key is None
    assert Settings(anthropic_api_key="sk-test").anthropic_api_key is not None
