"""SerpKite search tools for LangChain agents."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator

from ._utilities import SerpKiteAPIWrapper, normalize_endpoint

__all__ = ["SerpKiteSearch", "SerpKiteSearchInput", "SerpKiteSearchResults"]


class SerpKiteSearchInput(BaseModel):
    """Input for the SerpKite tools."""

    query: str = Field(description="The search query.")
    num: Optional[int] = Field(
        default=None, description="Number of results: 10 (default), or up to 100 for a deep search."
    )
    country: Optional[str] = Field(default=None, description="Two-letter country code, e.g. us, de, gb.")
    language: Optional[str] = Field(default=None, description="Language code, e.g. en, de, fr.")
    time: Optional[Literal["hour", "day", "week", "month", "year"]] = Field(
        default=None, description="Only results from the last hour, day, week, month or year."
    )


def _wrapper_from_kwargs(values: Any) -> Any:
    """Lets users pass ``api_key=`` / ``base_url=`` straight to the tool."""
    if isinstance(values, dict) and "api_wrapper" not in values:
        wrapper_kwargs = {k: values.pop(k) for k in ("api_key", "base_url") if k in values}
        if wrapper_kwargs:
            values["api_wrapper"] = SerpKiteAPIWrapper(**wrapper_kwargs)
    return values


class _SerpKiteTool(BaseTool):
    api_wrapper: SerpKiteAPIWrapper = Field(default_factory=SerpKiteAPIWrapper)
    """Holds the API key and HTTP clients (``SERPKITE_API_KEY`` by default)."""
    endpoint: str = "search"
    """Vertical to query: search, news, images, videos, maps, places, shopping, scholar, patents
    or autocomplete."""
    country: Optional[str] = None
    """Default country when the model doesn't pass one."""
    language: Optional[str] = None
    location: Optional[str] = None
    num: Optional[int] = None
    """Default number of results."""
    engine: Optional[Union[str, list[str]]] = None
    """Search providers allowed to answer (set by you, not the model): ``"google"`` (default),
    ``"auto"`` (fall back to other providers when Google is unavailable), one provider or a list.
    Falls back to ``api_wrapper.engine``."""

    args_schema: type[BaseModel] = SerpKiteSearchInput

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        return _wrapper_from_kwargs(values)

    @field_validator("endpoint")
    @classmethod
    def _check_endpoint(cls, value: str) -> str:
        return normalize_endpoint(value)

    def _params(
        self,
        num: Optional[int],
        country: Optional[str],
        language: Optional[str],
        time: Optional[str],
    ) -> dict[str, Any]:
        return {
            "num": num or self.num,
            "country": country or self.country,
            "language": language or self.language,
            "location": self.location,
            "time": time,
            "engine": self.engine,
        }


class SerpKiteSearch(_SerpKiteTool):
    """Google search via SerpKite, returning token-lean Markdown for LLM agents.

    Setup:
        .. code-block:: bash

            pip install -U langchain-serpkite
            export SERPKITE_API_KEY="skt_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_serpkite import SerpKiteSearch

            tool = SerpKiteSearch()  # or SerpKiteSearch(endpoint="news", country="de")

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "best espresso machine"})
    """

    name: str = "serpkite_search"
    description: str = (
        "Search Google and get the top results (titles, links, snippets, answer box) "
        "as Markdown. Use it for current events, facts you are unsure about and finding sources. "
        "Input is a search query."
    )

    def _run(
        self,
        query: str,
        num: Optional[int] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        time: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        params = self._params(num, country, language, time)
        return self.api_wrapper.markdown(query, self.endpoint, **params)

    async def _arun(
        self,
        query: str,
        num: Optional[int] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        time: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        params = self._params(num, country, language, time)
        return await self.api_wrapper.amarkdown(query, self.endpoint, **params)


class SerpKiteSearchResults(_SerpKiteTool):
    """Google search via SerpKite, returning the result list as dicts (title, link, snippet, …).

    Instantiate:
        .. code-block:: python

            from langchain_serpkite import SerpKiteSearchResults

            tool = SerpKiteSearchResults(max_results=5)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "langgraph checkpointer"})
            # [{"position": 1, "title": "...", "link": "https://...", "domain": "...", "snippet": "..."}]
    """

    name: str = "serpkite_search_results_json"
    description: str = (
        "Search Google and get the results as a JSON list of objects with title, link, domain and "
        "snippet. Use it when you need structured results or exact URLs. Input is a search query."
    )
    max_results: Optional[int] = None
    """Truncate the list to this many results."""

    def _shape(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        results = body.get("results") or []
        rows = [r for r in results if isinstance(r, dict)]
        return rows[: self.max_results] if self.max_results else rows

    def _run(
        self,
        query: str,
        num: Optional[int] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        time: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        params = self._params(num, country, language, time)
        return self._shape(self.api_wrapper.raw(query, self.endpoint, **params))

    async def _arun(
        self,
        query: str,
        num: Optional[int] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        time: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        params = self._params(num, country, language, time)
        return self._shape(await self.api_wrapper.araw(query, self.endpoint, **params))
