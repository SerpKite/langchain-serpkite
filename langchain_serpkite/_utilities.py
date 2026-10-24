"""SerpKiteAPIWrapper: holds the SerpKite clients shared by the tools, retriever and loader."""

from __future__ import annotations

from typing import Any, Optional, Union

from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr
from serpkite import AsyncSerpKite, SerpKite
from serpkite.types import AnswerResponse, ResearchTask, SearchResponse, TaskCreated, WebpageResponse

__all__ = ["SerpKiteAPIWrapper"]

VERTICALS = frozenset(
    {
        "search",
        "news",
        "images",
        "videos",
        "maps",
        "places",
        "shopping",
        "scholar",
        "patents",
        "autocomplete",
    }
)


def normalize_endpoint(endpoint: str) -> str:
    ep = endpoint.strip().lower()
    if ep not in VERTICALS:
        raise ValueError(f"unknown SerpKite endpoint {endpoint!r}; use one of {', '.join(sorted(VERTICALS))}")
    return ep


class SerpKiteAPIWrapper(BaseModel):
    """Thin wrapper around the ``serpkite`` SDK clients.

    The API key comes from ``api_key`` or the ``SERPKITE_API_KEY`` environment variable.

    Example:
        .. code-block:: python

            from langchain_serpkite import SerpKiteAPIWrapper

            wrapper = SerpKiteAPIWrapper(country="de")
            print(wrapper.markdown("beste espressomaschine"))
    """

    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "SERPKITE_API_KEY",
            error_message=(
                "No SerpKite API key. Pass api_key=... or set the SERPKITE_API_KEY environment variable "
                "(create a key at https://app.serpkite.com/keys)."
            ),
        )
    )
    """SerpKite API key (``skt_live_…``). Defaults to ``SERPKITE_API_KEY``."""
    base_url: Optional[str] = None
    """API origin; defaults to ``SERPKITE_BASE_URL`` or https://api.serpkite.com."""
    timeout: float = 60.0
    max_retries: int = 2
    country: Optional[str] = None
    """Default country code (e.g. ``us``) when a call doesn't pass one."""
    language: Optional[str] = None
    """Default interface language (e.g. ``en``)."""
    location: Optional[str] = None
    """Default free-text location, e.g. ``"Austin, Texas, United States"``."""
    engine: Optional[Union[str, list[str]]] = None
    """Default search providers allowed to answer: ``"google"`` (the API default, Google only),
    ``"auto"`` (fall back to other providers when Google is blocked or times out), one provider
    (``"brave"``) or a list (``["google", "brave"]``). ``meta.engine`` names the one that answered."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _client: Optional[SerpKite] = PrivateAttr(default=None)
    _async_client: Optional[AsyncSerpKite] = PrivateAttr(default=None)

    @property
    def client(self) -> SerpKite:
        """The sync SDK client (created on first use)."""
        if self._client is None:
            self._client = SerpKite(
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    @property
    def async_client(self) -> AsyncSerpKite:
        """The async SDK client (created on first use)."""
        if self._async_client is None:
            self._async_client = AsyncSerpKite(
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._async_client

    def _body(self, query: str, params: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "q": query,
            "country": self.country,
            "language": self.language,
            "location": self.location,
            "engine": self.engine,
        }
        body.update({k: v for k, v in params.items() if v is not None})
        return body

    # Sync -----------------------------------------------------------------

    def markdown(self, query: str, endpoint: str = "search", **params: Any) -> str:
        """Run ``query`` against ``endpoint`` and return the Markdown rendering."""
        body = {**self._body(query, params), "format": "markdown"}
        return str(self.client.request(normalize_endpoint(endpoint), body))

    def raw(self, query: str, endpoint: str = "search", **params: Any) -> dict[str, Any]:
        """Run ``query`` against ``endpoint`` and return the JSON body as a dict."""
        result: dict[str, Any] = self.client.request(normalize_endpoint(endpoint), self._body(query, params))
        return result

    def search(self, query: str, **params: Any) -> SearchResponse:
        """Typed ``/v1/search`` call."""
        return SearchResponse.model_validate(self.raw(query, "search", **params))

    def webpage(self, url: str, include_html: bool = False) -> WebpageResponse:
        """Typed ``/v1/webpage`` call."""
        return self.client.webpage(url, include_html=include_html)

    def _defaults(self, params: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        """Wrapper defaults (for ``keys``) under the call's params; ``None`` values dropped."""
        out: dict[str, Any] = {k: getattr(self, k) for k in keys}
        out.update({k: v for k, v in params.items() if v is not None})
        return {k: v for k, v in out.items() if v is not None}

    def answer(self, query: str, **params: Any) -> AnswerResponse:
        """Typed ``/v1/answer`` call: a cited answer from live Google results."""
        return self.client.answer(query, **self._defaults(params, _ANSWER_DEFAULTS))

    def research(self, input: str, **params: Any) -> TaskCreated:
        """Start a ``/v1/research`` task."""
        return self.client.research(input, **self._defaults(params, _RESEARCH_DEFAULTS))

    def get_research(self, id: str) -> ResearchTask:
        return self.client.get_research(id)

    def wait_research(self, id: str, timeout: float = 900.0) -> ResearchTask:
        """Poll a research task until it completed, failed or was canceled."""
        return self.client.wait_research(id, timeout=timeout)

    # Async ----------------------------------------------------------------

    async def amarkdown(self, query: str, endpoint: str = "search", **params: Any) -> str:
        body = {**self._body(query, params), "format": "markdown"}
        return str(await self.async_client.request(normalize_endpoint(endpoint), body))

    async def araw(self, query: str, endpoint: str = "search", **params: Any) -> dict[str, Any]:
        body = self._body(query, params)
        result: dict[str, Any] = await self.async_client.request(normalize_endpoint(endpoint), body)
        return result

    async def asearch(self, query: str, **params: Any) -> SearchResponse:
        return SearchResponse.model_validate(await self.araw(query, "search", **params))

    async def awebpage(self, url: str, include_html: bool = False) -> WebpageResponse:
        return await self.async_client.webpage(url, include_html=include_html)

    async def aanswer(self, query: str, **params: Any) -> AnswerResponse:
        return await self.async_client.answer(query, **self._defaults(params, _ANSWER_DEFAULTS))

    async def aresearch(self, input: str, **params: Any) -> TaskCreated:
        return await self.async_client.research(input, **self._defaults(params, _RESEARCH_DEFAULTS))

    async def aget_research(self, id: str) -> ResearchTask:
        return await self.async_client.get_research(id)

    async def await_research(self, id: str, timeout: float = 900.0) -> ResearchTask:
        return await self.async_client.wait_research(id, timeout=timeout)


_ANSWER_DEFAULTS = ("country", "language", "location", "engine")
_RESEARCH_DEFAULTS = ("country", "language")
