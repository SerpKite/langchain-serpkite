from __future__ import annotations

import json

import httpx
import pytest
import respx
from langchain_core.messages import ToolMessage
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_serpkite import SerpKiteAPIWrapper, SerpKiteSearch, SerpKiteSearchResults

from .fixtures import BASE, search_body


class TestSerpKiteSearchStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[SerpKiteSearch]:
        return SerpKiteSearch

    @property
    def tool_constructor_params(self) -> dict[str, object]:
        return {"api_key": "skt_live_param"}

    @property
    def tool_invoke_params_example(self) -> dict[str, object]:
        return {"query": "best espresso machine", "num": 10, "country": "us"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
        return {"SERPKITE_API_KEY": "skt_live_env"}, {}, {"name": "serpkite_search"}


class TestSerpKiteSearchResultsStandard(TestSerpKiteSearchStandard):
    @property
    def tool_constructor(self) -> type[SerpKiteSearchResults]:  # type: ignore[override]
        return SerpKiteSearchResults

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
        return {"SERPKITE_API_KEY": "skt_live_env"}, {}, {"name": "serpkite_search_results_json"}


@respx.mock
def test_search_tool_returns_markdown() -> None:
    route = respx.post(f"{BASE}/v1/search").mock(
        return_value=httpx.Response(200, text="# Results\n\n1. A", headers={"content-type": "text/markdown"})
    )
    tool = SerpKiteSearch(country="de", num=20)
    out = tool.invoke({"query": "espresso", "time": "week"})
    assert out == "# Results\n\n1. A"
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer skt_live_test"
    assert json.loads(req.content) == {
        "q": "espresso",
        "num": 20,
        "country": "de",
        "time": "week",
        "format": "markdown",
    }


@respx.mock
def test_search_tool_tool_call_gives_tool_message() -> None:
    respx.post(f"{BASE}/v1/news").mock(
        return_value=httpx.Response(200, text="## News", headers={"content-type": "text/markdown"})
    )
    tool = SerpKiteSearch(endpoint="news")
    msg = tool.invoke({"type": "tool_call", "id": "call_1", "name": tool.name, "args": {"query": "ai"}})
    assert isinstance(msg, ToolMessage)
    assert msg.content == "## News"
    assert msg.tool_call_id == "call_1"


@respx.mock
async def test_search_tool_async_and_wrapper_defaults() -> None:
    route = respx.post(f"{BASE}/v1/scholar").mock(
        return_value=httpx.Response(200, text="Answer", headers={"content-type": "text/markdown"})
    )
    wrapper = SerpKiteAPIWrapper(api_key="skt_live_w", country="fr", language="fr")
    tool = SerpKiteSearch(api_wrapper=wrapper, endpoint="scholar")
    assert await tool.ainvoke({"query": "q", "language": "en"}) == "Answer"
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer skt_live_w"
    assert json.loads(req.content) == {"q": "q", "country": "fr", "language": "en", "format": "markdown"}


@respx.mock
def test_tools_send_engine() -> None:
    route = respx.post(f"{BASE}/v1/search").mock(
        return_value=httpx.Response(200, text="# R", headers={"content-type": "text/markdown"})
    )
    SerpKiteSearch(engine="auto").invoke({"query": "q"})
    assert json.loads(route.calls.last.request.content) == {"q": "q", "engine": "auto", "format": "markdown"}

    wrapper = SerpKiteAPIWrapper(engine=["google", "brave"])
    SerpKiteSearch(api_wrapper=wrapper).invoke({"query": "q"})
    assert json.loads(route.calls.last.request.content) == {
        "q": "q",
        "engine": ["google", "brave"],
        "format": "markdown",
    }


@respx.mock
def test_results_tool_returns_dicts() -> None:
    route = respx.post(f"{BASE}/v1/search").mock(return_value=httpx.Response(200, json=search_body(10)))
    tool = SerpKiteSearchResults(max_results=3)
    rows = tool.invoke({"query": "espresso"})
    assert isinstance(rows, list)
    assert len(rows) == 3
    assert rows[0] == {
        "position": 1,
        "title": "Result 1",
        "link": "https://site1.example/page",
        "domain": "site1.example",
        "snippet": "Snippet 1",
    }
    assert json.loads(route.calls.last.request.content) == {"q": "espresso"}


@respx.mock
async def test_results_tool_async_tool_message_is_json() -> None:
    respx.post(f"{BASE}/v1/search").mock(return_value=httpx.Response(200, json=search_body(2)))
    tool = SerpKiteSearchResults()
    msg = await tool.ainvoke({"type": "tool_call", "id": "c", "name": tool.name, "args": {"query": "x"}})
    assert isinstance(msg, ToolMessage)
    assert isinstance(msg.content, str)
    assert json.loads(msg.content)[1]["title"] == "Result 2"


def test_invalid_endpoint() -> None:
    with pytest.raises(ValueError, match="unknown SerpKite endpoint"):
        SerpKiteSearch(endpoint="webpage")


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPKITE_API_KEY")
    with pytest.raises(ValueError, match="SERPKITE_API_KEY"):
        SerpKiteSearch()
