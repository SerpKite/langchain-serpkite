"""Live tests against the real API. Run with a real key:

SERPKITE_API_KEY=skt_live_... uv run pytest tests/integration_tests
"""

from __future__ import annotations

import os

import pytest
from langchain_tests.integration_tests import RetrieversIntegrationTests, ToolsIntegrationTests

from langchain_serpkite import SerpKiteRetriever, SerpKiteSearch, SerpKiteWebpageLoader

pytestmark = [
    pytest.mark.skipif(not os.environ.get("SERPKITE_API_KEY"), reason="SERPKITE_API_KEY not set"),
    pytest.mark.enable_socket,
]


class TestSerpKiteSearchLive(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[SerpKiteSearch]:
        return SerpKiteSearch

    @property
    def tool_constructor_params(self) -> dict[str, object]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, object]:
        return {"query": "langchain"}


class TestSerpKiteRetrieverLive(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[SerpKiteRetriever]:
        return SerpKiteRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, object]:
        return {"k": 3}

    @property
    def retriever_query_example(self) -> str:
        return "what is retrieval augmented generation"


def test_webpage_loader_live() -> None:
    docs = SerpKiteWebpageLoader("https://example.com").load()
    assert docs[0].page_content
