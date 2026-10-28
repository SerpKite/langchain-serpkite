from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import ToolMessage
from langchain_tests.unit_tests import ToolsUnitTests
from serpkite.types import AssistantResponse

from langchain_serpkite import SerpKiteAPIWrapper, SerpKiteAssistant

from .fixtures import BASE, meta


def assistant_body(endpoint: str = "chatgpt", **extra: Any) -> dict[str, Any]:
    return {
        "request": {"endpoint": endpoint, "prompt": "best crm"},
        "text": "HubSpot is popular [1].",
        "markdown": "HubSpot is popular [1](https://hubspot.com/).",
        "sources": [
            {
                "position": 1,
                "title": "HubSpot",
                "link": "https://hubspot.com/",
                "domain": "hubspot.com",
                "cited": True,
            },
            {"position": 2, "link": "https://g2.com/crm", "domain": "g2.com", "cited": False},
        ],
        "search_queries": ["best crm"],
        "products": [{"position": 1, "title": "HubSpot CRM", "price": "$0", "merchant": "hubspot.com"}],
        "mentions": [
            {"term": "hubspot.com", "mentioned": True, "cited": True, "sources": [1], "text_position": 0.0},
            {"term": "Pipedrive", "mentioned": False, "cited": False, "sources": [], "text_position": -1},
        ],
        "meta": {**meta(), "credits_used": 5},
        **extra,
    }


class TestSerpKiteAssistantStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[SerpKiteAssistant]:
        return SerpKiteAssistant

    @property
    def tool_constructor_params(self) -> dict[str, object]:
        return {"api_key": "skt_live_param", "assistant": "gemini"}

    @property
    def tool_invoke_params_example(self) -> dict[str, object]:
        return {"prompt": "best crm"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
        return {"SERPKITE_API_KEY": "skt_live_env"}, {}, {"name": "serpkite_chatgpt"}


@respx.mock
def test_chatgpt_tool_content_and_artifact() -> None:
    route = respx.post(f"{BASE}/v1/chatgpt").mock(return_value=httpx.Response(200, json=assistant_body()))
    tool = SerpKiteAssistant(track=["hubspot.com", "Pipedrive"], web_search=False, country="us")
    assert tool.name == "serpkite_chatgpt"
    assert "ChatGPT" in tool.description
    out = tool.invoke({"prompt": "best crm"})
    assert out.startswith("HubSpot is popular [1](https://hubspot.com/).")
    assert "[1] HubSpot - https://hubspot.com/" in out
    assert "[2] g2.com - https://g2.com/crm (listed, not cited)" in out
    assert "1. HubSpot CRM ($0, hubspot.com)" in out
    assert "- hubspot.com: named in the text; cited; sources 1" in out
    assert "- Pipedrive: not mentioned" in out
    assert json.loads(route.calls.last.request.content) == {
        "prompt": "best crm",
        "track": ["hubspot.com", "Pipedrive"],
        "country": "us",
        "web_search": False,
    }

    msg = tool.invoke({"type": "tool_call", "id": "1", "name": tool.name, "args": {"prompt": "q"}})
    assert isinstance(msg, ToolMessage) and isinstance(msg.artifact, AssistantResponse)
    assert msg.artifact.mentions and msg.artifact.mentions[0].term == "hubspot.com"


@pytest.mark.parametrize("assistant", ["gemini", "copilot", "ai_mode", "brave_ai"])
@respx.mock
async def test_other_assistants_async_with_wrapper_defaults(assistant: str) -> None:
    route = respx.post(f"{BASE}/v1/{assistant}").mock(
        return_value=httpx.Response(200, json=assistant_body(assistant, mentions=None, products=[]))
    )
    wrapper = SerpKiteAPIWrapper(country="de", language="de")
    tool = SerpKiteAssistant(api_wrapper=wrapper, assistant=assistant)
    assert tool.name == f"serpkite_{assistant}"
    out = await tool.ainvoke({"prompt": "beste crm"})
    assert "Sources:" in out and "Mentions:" not in out and "Products:" not in out
    assert json.loads(route.calls.last.request.content) == {
        "prompt": "beste crm",
        "country": "de",
        "language": "de",
    }


@respx.mock
def test_chatgpt_drops_the_wrapper_language() -> None:
    route = respx.post(f"{BASE}/v1/chatgpt").mock(return_value=httpx.Response(200, json=assistant_body()))
    SerpKiteAssistant(api_wrapper=SerpKiteAPIWrapper(api_key="k", language="de")).invoke({"prompt": "x"})
    assert json.loads(route.calls.last.request.content) == {"prompt": "x"}


def test_invalid_options() -> None:
    with pytest.raises(ValueError, match="web_search"):
        SerpKiteAssistant(api_key="k", assistant="gemini", web_search=True)
    with pytest.raises(ValueError, match="language"):
        SerpKiteAssistant(api_key="k", language="de")
    with pytest.raises(ValueError):
        SerpKiteAssistant(api_key="k", assistant="claude")
    tool = SerpKiteAssistant(api_key="k", assistant="brave_ai", name="brave", description="Ask Brave.")
    assert (tool.name, tool.description) == ("brave", "Ask Brave.")
