from __future__ import annotations

import json

import httpx
import pytest
import respx
from serpkite import NotFoundError, SerpKiteError

from langchain_serpkite import SerpKiteWebpageLoader

from .fixtures import BASE, webpage_body


@respx.mock
def test_load_documents() -> None:
    route = respx.post(f"{BASE}/v1/webpage").mock(
        return_value=httpx.Response(200, json=webpage_body("https://example.com"))
    )
    docs = SerpKiteWebpageLoader("https://example.com").load()
    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content.startswith("# Example Domain")
    assert doc.metadata == {
        "source": "https://example.com/",
        "requested_url": "https://example.com",
        "status_code": 200,
        "title": "Example Domain",
        "language": "en",
    }
    assert json.loads(route.calls.last.request.content) == {"url": "https://example.com"}


@respx.mock
def test_continue_on_failure() -> None:
    respx.post(f"{BASE}/v1/webpage").mock(
        side_effect=[
            httpx.Response(503, json={"error": {"code": "upstream_error", "message": "fetch failed"}}),
            httpx.Response(200, json=webpage_body("https://b.example")),
        ]
    )
    loader = SerpKiteWebpageLoader(
        ["https://a.example", "https://b.example"], continue_on_failure=True, api_key="skt_live_x"
    )
    loader.api_wrapper.max_retries = 0
    docs = loader.load()
    assert [d.metadata["requested_url"] for d in docs] == ["https://b.example"]


@respx.mock
def test_raises_by_default() -> None:
    respx.post(f"{BASE}/v1/webpage").mock(
        return_value=httpx.Response(404, json={"error": {"code": "not_found", "message": "gone"}})
    )
    with pytest.raises(NotFoundError):
        SerpKiteWebpageLoader(["https://a.example"]).load()


@respx.mock
async def test_alazy_load_with_html() -> None:
    body = webpage_body("https://example.com")
    body["html"] = "<h1>Example Domain</h1>"
    route = respx.post(f"{BASE}/v1/webpage").mock(return_value=httpx.Response(200, json=body))
    loader = SerpKiteWebpageLoader(["https://example.com", "https://example.com"], include_html=True)
    docs = [doc async for doc in loader.alazy_load()]
    assert len(docs) == 2
    assert docs[0].metadata["html"] == "<h1>Example Domain</h1>"
    assert json.loads(route.calls.last.request.content) == {
        "url": "https://example.com",
        "include_html": True,
    }
    assert issubclass(NotFoundError, SerpKiteError)
