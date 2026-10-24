from __future__ import annotations

import json
from typing import Any

import httpx
import respx
from langchain_core.messages import ToolMessage
from langchain_tests.unit_tests import ToolsUnitTests
from serpkite.types import AnswerResponse, ResearchTask

from langchain_serpkite import SerpKiteAnswer, SerpKiteAPIWrapper, SerpKiteResearch

from .fixtures import BASE, meta

TID = "01926a3e-7b2c-7c4e-9f1a-2b3c4d5e6f70"
CITATION = {
    "index": 1,
    "title": "SERP",
    "link": "https://en.wikipedia.org/wiki/SERP",
    "domain": "en.wikipedia.org",
}


def answer_body(**extra: Any) -> dict[str, Any]:
    return {
        "request": {"endpoint": "answer", "engine": "google", "q": "what is a serp"},
        "answer": "A SERP is a results page [1].",
        "citations": [CITATION],
        "results": [CITATION],
        "meta": {**meta(), "credits": {"search": 1, "answer": 4}},
        **extra,
    }


def research_task(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": TID,
        "kind": "research",
        "status": status,
        "created_at": "2026-10-24T12:00:00Z",
        "credits_reserved": 75,
        "credits_used": 0,
        **extra,
    }


CREATED = {
    "id": TID,
    "kind": "research",
    "status": "queued",
    "created_at": "2026-10-24T12:00:00Z",
    "poll_url": f"{BASE}/v1/research/{TID}",
    "credits_reserved": 75,
    "webhook_url": None,
}
REPORT = {
    "input": "vector dbs",
    "effort": "lite",
    "report": "# Vector databases\n\npgvector is popular [1].",
    "citations": [CITATION],
    "sources": [{**CITATION, "query": "vector databases"}],
    "queries": ["vector databases"],
}


class TestSerpKiteAnswerStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[SerpKiteAnswer]:
        return SerpKiteAnswer

    @property
    def tool_constructor_params(self) -> dict[str, object]:
        return {"api_key": "skt_live_param"}

    @property
    def tool_invoke_params_example(self) -> dict[str, object]:
        return {"query": "what is a serp"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
        return {"SERPKITE_API_KEY": "skt_live_env"}, {}, {"name": "serpkite_answer"}


@respx.mock
def test_answer_tool_content_and_artifact() -> None:
    route = respx.post(f"{BASE}/v1/answer").mock(return_value=httpx.Response(200, json=answer_body()))
    tool = SerpKiteAnswer(depth="fast", country="us", num_sources=4)
    out = tool.invoke({"query": "what is a serp"})
    assert out.startswith("A SERP is a results page [1].")
    assert "[1] SERP - https://en.wikipedia.org/wiki/SERP" in out
    # num_sources only applies to depth=standard
    assert json.loads(route.calls.last.request.content) == {
        "q": "what is a serp",
        "depth": "fast",
        "country": "us",
    }

    msg = tool.invoke({"type": "tool_call", "id": "1", "name": tool.name, "args": {"query": "q"}})
    assert isinstance(msg, ToolMessage) and isinstance(msg.artifact, AnswerResponse)


@respx.mock
async def test_answer_tool_async_and_wrapper_defaults() -> None:
    route = respx.post(f"{BASE}/v1/answer").mock(
        return_value=httpx.Response(200, json=answer_body(answer=None, citations=[]))
    )
    wrapper = SerpKiteAPIWrapper(country="de", language="de")
    out = await SerpKiteAnswer(api_wrapper=wrapper, num_sources=2, include_domains=["wikipedia.org"]).ainvoke(
        {"query": "q"}
    )
    assert out == "No answer could be found in the search results."
    assert json.loads(route.calls.last.request.content) == {
        "q": "q",
        "depth": "standard",
        "num_sources": 2,
        "include_domains": ["wikipedia.org"],
        "country": "de",
        "language": "de",
    }


@respx.mock
def test_research_tool_waits_for_the_report() -> None:
    create = respx.post(f"{BASE}/v1/research").mock(return_value=httpx.Response(202, json=CREATED))
    respx.get(f"{BASE}/v1/research/{TID}").mock(
        side_effect=[
            httpx.Response(200, json=research_task("running", progress={"round": 1})),
            httpx.Response(200, json=research_task("completed", credits_used=75, result=REPORT)),
        ]
    )
    tool = SerpKiteResearch(effort="lite", include_domains=["github.com"])
    msg = tool.invoke({"type": "tool_call", "id": "1", "name": tool.name, "args": {"query": "vector dbs"}})
    assert isinstance(msg, ToolMessage) and isinstance(msg.artifact, ResearchTask)
    assert str(msg.content).startswith("# Vector databases")
    assert "Sources:\n[1] SERP" in str(msg.content)
    assert json.loads(create.calls.last.request.content) == {
        "input": "vector dbs",
        "effort": "lite",
        "include_domains": ["github.com"],
    }


@respx.mock
async def test_research_tool_failed_and_timeout() -> None:
    respx.post(f"{BASE}/v1/research").mock(return_value=httpx.Response(202, json=CREATED))
    respx.get(f"{BASE}/v1/research/{TID}").mock(
        return_value=httpx.Response(
            200, json=research_task("failed", error={"code": "ai_unavailable", "message": "model failed"})
        )
    )
    out = await SerpKiteResearch().ainvoke({"query": "x"})
    assert out == "Research failed: ai_unavailable: model failed"

    respx.get(f"{BASE}/v1/research/{TID}").mock(
        return_value=httpx.Response(200, json=research_task("running"))
    )
    out = SerpKiteResearch(timeout=0).invoke({"query": "x"})
    assert out == f"Research {TID} is running; check again later."
