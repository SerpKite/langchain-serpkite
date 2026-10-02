from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import respx
from langchain_core.documents import Document
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_serpkite import SerpKiteRetriever

from .fixtures import BASE, search_body


class TestSerpKiteRetrieverStandard(RetrieversIntegrationTests):
    """LangChain's standard retriever suite, run against a mocked API."""

    @pytest.fixture(autouse=True)
    def _mock_api(self) -> Iterator[None]:
        with respx.mock:
            respx.post(f"{BASE}/v1/search").mock(return_value=httpx.Response(200, json=search_body(10)))
            yield

    @property
    def retriever_constructor(self) -> type[SerpKiteRetriever]:
        return SerpKiteRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, object]:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        return "what is a vector database"


@respx.mock
def test_documents_from_content_and_snippets() -> None:
    route = respx.post(f"{BASE}/v1/search").mock(
        return_value=httpx.Response(200, json=search_body(10, content_for=2))
    )
    retriever = SerpKiteRetriever(k=5, include_content=2, country="us", language="en")
    docs = retriever.invoke("hnsw")
    assert len(docs) == 5
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content.startswith("# Page 1")
    assert docs[2].page_content == "Snippet 3"
    assert docs[0].metadata == {
        "title": "Result 1",
        "link": "https://site1.example/page",
        "source": "https://site1.example/page",
        "position": 1,
        "domain": "site1.example",
        "snippet": "Snippet 1",
    }
    assert json.loads(route.calls.last.request.content) == {
        "q": "hnsw",
        "include_content": 2,
        "country": "us",
        "language": "en",
    }


@respx.mock
def test_k_above_ten_requests_depth_and_caps_include_content() -> None:
    route = respx.post(f"{BASE}/v1/search").mock(return_value=httpx.Response(200, json=search_body(30)))
    docs = SerpKiteRetriever(include_content=5).invoke("q", k=25)
    assert len(docs) == 25
    assert json.loads(route.calls.last.request.content) == {"q": "q", "num": 25, "include_content": 5}

    SerpKiteRetriever(k=1, include_content=5).invoke("q")
    assert json.loads(route.calls.last.request.content) == {"q": "q", "include_content": 1}


@respx.mock
def test_engine_param_and_metadata() -> None:
    body = search_body(3)
    body["meta"]["engine"] = "brave"
    body["meta"]["route"] = [
        {"provider": "google", "outcome": "blocked", "ms": 800},
        {"provider": "brave", "outcome": "ok", "ms": 400},
    ]
    route = respx.post(f"{BASE}/v1/search").mock(return_value=httpx.Response(200, json=body))
    docs = SerpKiteRetriever(k=2, engine="auto").invoke("q")
    assert json.loads(route.calls.last.request.content) == {"q": "q", "engine": "auto"}
    assert [d.metadata["engine"] for d in docs] == ["brave", "brave"]

    SerpKiteRetriever(k=2).invoke("q", engine=["google", "brave"])
    assert json.loads(route.calls.last.request.content) == {"q": "q", "engine": ["google", "brave"]}


@respx.mock
def test_consensus_sources_metadata() -> None:
    body = search_body(2)
    body["meta"]["engine"] = "consensus"
    body["results"][0]["sources"] = ["google", "brave"]
    route = respx.post(f"{BASE}/v1/search").mock(return_value=httpx.Response(200, json=body))
    docs = SerpKiteRetriever(k=2, engine="consensus").invoke("q")
    assert json.loads(route.calls.last.request.content) == {"q": "q", "engine": "consensus"}
    assert docs[0].metadata["sources"] == ["google", "brave"]
    assert docs[0].metadata["engine"] == "consensus"
    assert "sources" not in docs[1].metadata


@respx.mock
async def test_async_retriever() -> None:
    respx.post(f"{BASE}/v1/search").mock(return_value=httpx.Response(200, json=search_body(4)))
    docs = await SerpKiteRetriever(k=3, time="day").ainvoke("q")
    assert [d.metadata["position"] for d in docs] == [1, 2, 3]


def test_validation() -> None:
    with pytest.raises(ValueError):
        SerpKiteRetriever(include_content=6)
    with pytest.raises(ValueError):
        SerpKiteRetriever(k=0)
