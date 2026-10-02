"""SerpKiteRetriever: Google results as LangChain Documents."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator
from serpkite.types import OrganicResult, SearchResponse

from ._utilities import SerpKiteAPIWrapper

__all__ = ["SerpKiteRetriever"]

MAX_INCLUDE_CONTENT = 5


class SerpKiteRetriever(BaseRetriever):
    """Retrieve Google web results as Documents via SerpKite.

    ``page_content`` is the result page as Markdown when ``include_content`` covered it,
    otherwise the result snippet. Metadata carries ``title``, ``link`` (also as ``source``),
    ``position``, ``domain`` and ``engine`` (the search provider that answered, e.g. ``google``).

    Setup:
        .. code-block:: bash

            pip install -U langchain-serpkite
            export SERPKITE_API_KEY="skt_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_serpkite import SerpKiteRetriever

            retriever = SerpKiteRetriever(k=5, include_content=2, country="us")

    Usage:
        .. code-block:: python

            docs = retriever.invoke("how does HNSW indexing work")
            docs = retriever.invoke("...", k=3)  # override k per call
    """

    api_wrapper: SerpKiteAPIWrapper = Field(default_factory=SerpKiteAPIWrapper)
    k: int = Field(default=10, ge=1, le=100)
    """Number of documents to return. Above 10, a deep search is used (7 credits for 100)."""
    include_content: int = Field(default=0, ge=0, le=MAX_INCLUDE_CONTENT)
    """Fetch the top N (0-5) result pages as Markdown for ``page_content`` (+1 credit each)."""
    country: Optional[str] = None
    language: Optional[str] = None
    location: Optional[str] = None
    time: Optional[Literal["hour", "day", "week", "month", "year"]] = None
    """Only results from the last hour/day/week/month/year."""
    engine: Optional[Union[str, list[str]]] = None
    """Search providers allowed to answer: ``"google"`` (default), ``"auto"`` (fall back to other
    providers when Google is blocked or times out), one provider or a list."""

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        if isinstance(values, dict) and "api_wrapper" not in values:
            wrapper_kwargs = {k: values.pop(k) for k in ("api_key", "base_url") if k in values}
            if wrapper_kwargs:
                values["api_wrapper"] = SerpKiteAPIWrapper(**wrapper_kwargs)
        return values

    def _params(self, kwargs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        k = int(kwargs.pop("k", self.k))
        include = int(kwargs.pop("include_content", self.include_content))
        params: dict[str, Any] = {
            "num": k if k > 10 else None,
            "include_content": min(include, k) or None,
            "country": self.country,
            "language": self.language,
            "location": self.location,
            "time": self.time,
            "engine": self.engine,
        }
        params.update(kwargs)
        return k, params

    @staticmethod
    def _to_document(result: OrganicResult, engine: Optional[str] = None) -> Document:
        metadata: dict[str, Any] = {
            "title": result.title,
            "link": result.link,
            "source": result.link,
            "position": result.position,
            "domain": result.domain,
        }
        for key in ("snippet", "date", "displayed_link"):
            value = getattr(result, key, None)
            if value:
                metadata[key] = value
        if engine:
            metadata["engine"] = engine
        sources = getattr(result, "sources", None)
        if sources:  # engine="consensus": the providers that returned this result
            metadata["sources"] = list(sources)
        content = result.content or result.snippet or result.title
        return Document(page_content=content, metadata=metadata)

    def _documents(self, res: SearchResponse, k: int) -> list[Document]:
        return [self._to_document(r, res.meta.engine) for r in res.results[:k]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k, params = self._params(kwargs)
        return self._documents(self.api_wrapper.search(query, **params), k)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k, params = self._params(kwargs)
        return self._documents(await self.api_wrapper.asearch(query, **params), k)
