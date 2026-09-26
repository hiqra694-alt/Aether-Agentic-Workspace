import json
import sys
import os
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import providers.groq as groq_module
from providers.groq import GroqProvider


@pytest.fixture(autouse=True)
def _stub_settings(monkeypatch):
    """GroqProvider.__init__ calls get_settings() and only reads
    .groq_api_key off the result, so a bare stub is enough to satisfy the
    constructor guard without touching .env or constructing a real Settings
    (which would read .env again)."""
    stub = SimpleNamespace(groq_api_key="test-key")
    monkeypatch.setattr(groq_module, "get_settings", lambda: stub)


def _bad_request_error(failed_generation):
    """
    Builds a real openai.BadRequestError shaped exactly like what the SDK
    raises for Groq's `tool_use_failed` response: `.body` is the *inner*
    `error` dict (openai._client._make_status_error unwraps
    {"error": {...}} down to just that dict), which is where
    `failed_generation` lives.
    """
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {
        "message": "Failed to call a function. Please adjust your prompt and try again.",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": failed_generation,
    }
    return BadRequestError("Failed to call a function.", response=response, body=body)


def _chunk(content=None):
    """Mimics one streamed ChatCompletionChunk: only `.choices[0].delta.content`
    is read by GroqProvider for these tests (no native tool_calls delta)."""
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStream:
    """Minimal async-iterable standing in for the AsyncOpenAI streaming response."""
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for c in self._chunks:
            yield c


TOOLS = [
    {"type": "function", "function": {"name": "duckduckgo_search", "parameters": {}}},
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
]


def _make_provider(monkeypatch, response_chunks=None, raise_error=None):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = GroqProvider()

    captured_kwargs = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        if raise_error is not None:
            raise raise_error
        return _FakeStream(response_chunks)

    provider.client.chat.completions.create = fake_create
    return provider, captured_kwargs


@pytest.mark.asyncio
async def test_tool_choice_auto_bound_when_tools_present(monkeypatch):
    provider, captured_kwargs = _make_provider(monkeypatch, [_chunk("Hello there.")])

    async for _ in provider.stream_response([], tools=TOOLS):
        pass

    assert captured_kwargs["tools"] == TOOLS
    assert captured_kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_no_tool_choice_when_no_tools(monkeypatch):
    provider, captured_kwargs = _make_provider(monkeypatch, [_chunk("Hello there.")])

    async for _ in provider.stream_response([], tools=[]):
        pass

    assert "tool_choice" not in captured_kwargs
    assert "tools" not in captured_kwargs


@pytest.mark.asyncio
async def test_known_tool_tag_suppressed_and_yielded_as_text_tool_call(monkeypatch):
    """
    The exact regression from production: llama-3.3-70b-versatile emits
    <duckduckgo_search>{"search_query": "..."}</duckduckgo_search> as plain
    content instead of a native tool_calls delta. It must never appear in a
    yielded string chunk, and must instead surface as a structured
    text_tool_call event with the name/arguments extracted.
    """
    raw_tag = '<duckduckgo_search>{"search_query": "RAG and CAG meaning"}</duckduckgo_search>'
    provider, _ = _make_provider(monkeypatch, [_chunk(raw_tag)])

    events = []
    async for chunk in provider.stream_response([], tools=TOOLS):
        events.append(chunk)

    text_chunks = [e for e in events if isinstance(e, str)]
    tool_events = [e for e in events if isinstance(e, dict) and e.get("type") == "text_tool_call"]

    assert not any("duckduckgo_search" in t or "<" in t for t in text_chunks), text_chunks
    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "duckduckgo_search"
    assert json.loads(tool_events[0]["arguments"]) == {"search_query": "RAG and CAG meaning"}


@pytest.mark.asyncio
async def test_known_tool_tag_split_across_chunks_still_suppressed(monkeypatch):
    """The tag can arrive token-by-token across many small deltas; the
    buffering/hold logic must still assemble and catch it rather than
    leaking a partial `<duckduck` prefix to the user."""
    raw_tag = '<get_weather>{"city": "Lahore"}</get_weather>'
    pieces = [raw_tag[i:i + 3] for i in range(0, len(raw_tag), 3)]
    provider, _ = _make_provider(monkeypatch, [_chunk(p) for p in pieces] + [_chunk(" All done.")])

    events = []
    async for chunk in provider.stream_response([], tools=TOOLS):
        events.append(chunk)

    text_chunks = [e for e in events if isinstance(e, str)]
    tool_events = [e for e in events if isinstance(e, dict) and e.get("type") == "text_tool_call"]

    joined_text = "".join(text_chunks)
    assert "<" not in joined_text and "get_weather" not in joined_text
    assert "All done." in joined_text
    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "get_weather"
    assert json.loads(tool_events[0]["arguments"]) == {"city": "Lahore"}


@pytest.mark.asyncio
async def test_registered_tag_not_offered_this_turn_is_suppressed_but_not_executed(monkeypatch):
    """
    A tag naming a real, registered tool that just wasn't offered this
    specific turn (e.g. a post-tool synthesis turn, or -- as here -- a turn
    that only offered duckduckgo_search while the model hallucinates
    get_weather) must never leak the raw tag to the browser, and must never
    be executed either -- "never invoke a tool outside what was explicitly
    offered this turn" applies even for a tool that's real elsewhere.
    """
    provider, _ = _make_provider(monkeypatch, [_chunk('<get_weather>{"city": "Lahore"}</get_weather> Anyway,'), _chunk(" here's the answer.")])

    events = []
    async for chunk in provider.stream_response([], tools=[TOOLS[0]]):  # only duckduckgo_search offered
        events.append(chunk)

    text_chunks = [e for e in events if isinstance(e, str)]
    tool_events = [e for e in events if isinstance(e, dict) and e.get("type") == "text_tool_call"]

    joined_text = "".join(text_chunks)
    assert "<get_weather>" not in joined_text
    assert "here's the answer." in joined_text
    assert tool_events == []


@pytest.mark.asyncio
async def test_synthesis_turn_with_no_tools_still_suppresses_leaked_tag(monkeypatch):
    """
    Regression coverage for the synthesis-pass leak gap: on a turn where
    ChatService intentionally passes tools=[] (pure synthesis, after a tool
    has already run once), a hallucinated tag for a real tool must still
    never reach the browser as raw text, even though nothing is executable
    this turn.
    """
    provider, _ = _make_provider(monkeypatch, [_chunk('<search_knowledge_base>{"query": "x"}</search_knowledge_base>The answer is 42.')])

    events = []
    async for chunk in provider.stream_response([], tools=[]):
        events.append(chunk)

    text_chunks = [e for e in events if isinstance(e, str)]
    tool_events = [e for e in events if isinstance(e, dict) and e.get("type") == "text_tool_call"]

    joined_text = "".join(text_chunks)
    assert "<search_knowledge_base>" not in joined_text
    assert "The answer is 42." in joined_text
    assert tool_events == []


@pytest.mark.asyncio
async def test_never_registered_tag_is_suppressed_and_discarded(monkeypatch):
    """
    Task-specified case: a tag naming something that was never a real tool
    at all (e.g. <non_existent_tool>) must still be stripped from the
    visible stream -- any well-formed <tag>{json}</tag> pair is a leaked
    tool-call artifact regardless of whether the name matches a registered
    tool -- but obviously can never be executed.
    """
    raw_tag = '<non_existent_tool>{"foo": "bar"}</non_existent_tool>'
    provider, _ = _make_provider(monkeypatch, [_chunk(raw_tag + " Plain answer follows.")])

    events = []
    async for chunk in provider.stream_response([], tools=TOOLS):
        events.append(chunk)

    text_chunks = [e for e in events if isinstance(e, str)]
    tool_events = [e for e in events if isinstance(e, dict) and e.get("type") == "text_tool_call"]

    joined_text = "".join(text_chunks)
    assert "non_existent_tool" not in joined_text
    assert "Plain answer follows." in joined_text
    assert tool_events == []


@pytest.mark.asyncio
async def test_html_tag_with_plain_text_content_not_stripped(monkeypatch):
    """
    False-positive guard: ordinary prose discussing an HTML/XML tag (e.g.
    explaining `<div>hello</div>` in a coding answer) must survive intact
    -- the generic tag interceptor only matches when the wrapped content is
    a JSON object, which real prose essentially never is.
    """
    provider, _ = _make_provider(monkeypatch, [_chunk("Use <div>hello</div> to wrap it.")])

    events = []
    async for chunk in provider.stream_response([], tools=TOOLS):
        events.append(chunk)

    text_chunks = [e for e in events if isinstance(e, str)]
    assert "".join(text_chunks) == "Use <div>hello</div> to wrap it."


@pytest.mark.asyncio
async def test_ordinary_text_unaffected(monkeypatch):
    provider, _ = _make_provider(monkeypatch, [_chunk("The capital of France "), _chunk("is Paris.")])

    events = []
    async for chunk in provider.stream_response([], tools=TOOLS):
        events.append(chunk)

    text_chunks = [e for e in events if isinstance(e, str)]
    assert "".join(text_chunks) == "The capital of France is Paris."


@pytest.mark.asyncio
async def test_failed_generation_truncated_function_eq_tag_recovered(monkeypatch):
    """
    The exact production regression: Groq rejects the request outright
    (400 tool_use_failed) before any content streams, because
    llama-3.3-70b-versatile's own tool call generation was truncated
    mid-argument. The raw attempt -- preserved in the error body's
    `failed_generation` field -- must be recovered into a real tool call
    instead of the caller ever seeing the exception (which previously
    forced a retry with tool-calling disabled).
    """
    error = _bad_request_error('<function=get_weather>{"city": "San Francisco"')
    provider, _ = _make_provider(monkeypatch, raise_error=error)

    events = []
    async for chunk in provider.stream_response([], tools=TOOLS):
        events.append(chunk)

    tool_call_events = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_calls"]
    assert len(tool_call_events) == 1
    calls = tool_call_events[0]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "San Francisco"}


@pytest.mark.asyncio
async def test_failed_generation_bare_name_envelope_recovered(monkeypatch):
    """The other shape Groq's `failed_generation` can take: a clean bare
    {"name": ..., "arguments": {...}} envelope with no XML-style tag at all."""
    error = _bad_request_error(json.dumps({"name": "duckduckgo_search", "arguments": {"search_query": "coffee shops near me"}}))
    provider, _ = _make_provider(monkeypatch, raise_error=error)

    events = []
    async for chunk in provider.stream_response([], tools=TOOLS):
        events.append(chunk)

    tool_call_events = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_calls"]
    assert len(tool_call_events) == 1
    calls = tool_call_events[0]["tool_calls"]
    assert calls[0]["function"]["name"] == "duckduckgo_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"search_query": "coffee shops near me"}


@pytest.mark.asyncio
async def test_failed_generation_tool_not_offered_this_turn_not_recovered(monkeypatch):
    """A recovered call must still respect "never invoke a tool outside
    what was explicitly offered this turn" -- same rule the text-tag
    fallback enforces -- so an unrecoverable error still propagates as
    a real exception rather than silently executing an unoffered tool."""
    error = _bad_request_error('<function=get_weather>{"city": "Lahore"}</function>')
    provider, _ = _make_provider(monkeypatch, raise_error=error)

    with pytest.raises(BadRequestError):
        async for _ in provider.stream_response([], tools=[TOOLS[0]]):  # only duckduckgo_search offered
            pass


@pytest.mark.asyncio
async def test_failed_generation_unrecoverable_garbage_reraises(monkeypatch):
    """When nothing usable can be salvaged (e.g. the model's attempt was
    just prose, not a call at all), the original exception must still
    propagate so ChatService's existing degrade-gracefully retry runs --
    recovery must never mask a genuinely unrecoverable failure."""
    error = _bad_request_error("maybe")
    provider, _ = _make_provider(monkeypatch, raise_error=error)

    with pytest.raises(BadRequestError):
        async for _ in provider.stream_response([], tools=TOOLS):
            pass
