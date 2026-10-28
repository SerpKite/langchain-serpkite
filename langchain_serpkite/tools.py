"""SerpKite search tools for LangChain agents."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from serpkite.types import AnswerResponse, AssistantResponse, ResearchTask

from ._utilities import Assistant, SerpKiteAPIWrapper, normalize_endpoint

__all__ = [
    "SerpKiteAnswer",
    "SerpKiteAnswerInput",
    "SerpKiteAssistant",
    "SerpKiteAssistantInput",
    "SerpKiteResearch",
    "SerpKiteResearchInput",
    "SerpKiteSearch",
    "SerpKiteSearchInput",
    "SerpKiteSearchResults",
]


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


# ── AI: answer and research ────────────────────────────────────────────────


class SerpKiteAnswerInput(BaseModel):
    """Input for :class:`SerpKiteAnswer`."""

    query: str = Field(description="The question to answer from live Google results.")


def _with_sources(text: Optional[str], citations: list[Any], empty: str) -> str:
    """The text followed by its numbered sources."""
    lines = [text or empty]
    if citations:
        lines += ["", "Sources:"]
        lines += [f"[{c.index}] {c.title} - {c.link}" for c in citations]
    return "\n".join(lines)


def format_answer(res: AnswerResponse) -> str:
    """The answer (Markdown with ``[n]`` markers) followed by its numbered sources."""
    return _with_sources(res.answer, res.citations, "No answer could be found in the search results.")


class SerpKiteAnswer(BaseTool):
    """A cited answer grounded on live Google results, via SerpKite ``/v1/answer``.

    Returns the answer (Markdown with ``[n]`` citation markers) followed by the numbered
    sources. ``depth="fast"`` answers from the result page; ``"standard"`` (default) also reads
    the top ``num_sources`` pages (prices: https://serpkite.com/pricing). ``response_format`` is
    ``content_and_artifact``: the artifact is the full :class:`serpkite.types.AnswerResponse`
    (citations, results, ``meta.credits``, ``meta.llm``).

    Setup:
        .. code-block:: bash

            pip install -U langchain-serpkite
            export SERPKITE_API_KEY="skt_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_serpkite import SerpKiteAnswer

            tool = SerpKiteAnswer(depth="fast")

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "what is the latest stable rust version"})
    """

    name: str = "serpkite_answer"
    description: str = (
        "Answer a question from live Google results. Returns a concise answer with [n] citations "
        "followed by the numbered source links. Use it for current facts that need sources. "
        "Input is a question."
    )
    args_schema: type[BaseModel] = SerpKiteAnswerInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    api_wrapper: SerpKiteAPIWrapper = Field(default_factory=SerpKiteAPIWrapper)
    """Holds the API key and HTTP clients (``SERPKITE_API_KEY`` by default)."""
    depth: Literal["fast", "standard"] = "standard"
    num_sources: Optional[int] = Field(default=None, ge=1, le=5)
    """Pages read with ``depth="standard"`` (1-5, server default 3)."""
    include_domains: Optional[list[str]] = None
    exclude_domains: Optional[list[str]] = None
    country: Optional[str] = None
    """Defaults to ``api_wrapper.country``."""
    language: Optional[str] = None
    time: Optional[Literal["hour", "day", "week", "month", "year"]] = None
    """Only answer from results of the last hour, day, week, month or year."""

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        return _wrapper_from_kwargs(values)

    def _params(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "num_sources": self.num_sources if self.depth == "standard" else None,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "country": self.country,
            "language": self.language,
            "time": self.time,
        }

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> tuple[str, AnswerResponse]:
        res = self.api_wrapper.answer(query, **self._params())
        return format_answer(res), res

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> tuple[str, AnswerResponse]:
        res = await self.api_wrapper.aanswer(query, **self._params())
        return format_answer(res), res


class SerpKiteResearchInput(BaseModel):
    """Input for :class:`SerpKiteResearch`."""

    query: str = Field(description="The research task or question, in plain words.")


def format_research(task: ResearchTask) -> str:
    """The report followed by its numbered sources, or the task's state when it has none."""
    result = task.result
    report = getattr(result, "report", None) if result is not None else None
    if task.status == "completed" and report:
        return _with_sources(report, list(getattr(result, "citations", None) or []), "")
    if task.error is not None:
        return f"Research {task.status}: {task.error.code}: {task.error.message}"
    return f"Research {task.id} is {task.status}; check again later."


class SerpKiteResearch(BaseTool):
    """Deep research via SerpKite ``/v1/research``: planned Google searches, page reading and a
    cited Markdown report.

    The tool starts the task and waits for the report (one to several minutes, up to
    ``timeout`` seconds), then returns it followed by the numbered sources. ``effort`` (``lite``,
    ``standard``, ``deep``) sets how much is searched and read, at a flat price per task charged
    only when the report is written (https://serpkite.com/pricing). ``response_format`` is
    ``content_and_artifact``: the artifact is the :class:`serpkite.types.ResearchTask`.

    Instantiate:
        .. code-block:: python

            from langchain_serpkite import SerpKiteResearch

            tool = SerpKiteResearch(effort="lite")

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "compare managed Postgres vector search options"})
    """

    name: str = "serpkite_research"
    description: str = (
        "Research a question in depth: plans several Google searches, reads the sources and "
        "writes a Markdown report with [n] citations, followed by the numbered source links. "
        "Takes one to several minutes; use serpkite_answer for quick facts. Input is the research "
        "task in plain words."
    )
    args_schema: type[BaseModel] = SerpKiteResearchInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    api_wrapper: SerpKiteAPIWrapper = Field(default_factory=SerpKiteAPIWrapper)
    effort: Literal["lite", "standard", "deep"] = "standard"
    include_domains: Optional[list[str]] = None
    exclude_domains: Optional[list[str]] = None
    country: Optional[str] = None
    language: Optional[str] = None
    timeout: float = 900.0
    """Seconds to wait for the report; the tool then returns the task's state (it keeps running
    server-side)."""

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        return _wrapper_from_kwargs(values)

    def _params(self) -> dict[str, Any]:
        return {
            "effort": self.effort,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "country": self.country,
            "language": self.language,
        }

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> tuple[str, ResearchTask]:
        task = self.api_wrapper.research(query, **self._params())
        try:
            done = self.api_wrapper.wait_research(str(task.id), timeout=self.timeout)
        except TimeoutError:
            done = self.api_wrapper.get_research(str(task.id))
        return format_research(done), done

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> tuple[str, ResearchTask]:
        task = await self.api_wrapper.aresearch(query, **self._params())
        try:
            done = await self.api_wrapper.await_research(str(task.id), timeout=self.timeout)
        except TimeoutError:
            done = await self.api_wrapper.aget_research(str(task.id))
        return format_research(done), done


# ── AI assistants: ChatGPT, Gemini, Copilot, AI Mode, Brave AI ───────────────


class SerpKiteAssistantInput(BaseModel):
    """Input for :class:`SerpKiteAssistant`."""

    prompt: str = Field(description="The message to the assistant, as a user would type it.")


_ASSISTANT_INFO: dict[str, tuple[str, str]] = {
    "chatgpt": (
        "ChatGPT",
        "Ask ChatGPT (chatgpt.com, logged out, searching the web) and get its answer with the sources "
        "it cited and the products and places it showed: what ChatGPT tells people. Takes 10-90 s.",
    ),
    "gemini": (
        "Gemini",
        "Ask Google Gemini (gemini.google.com, logged out) and get its answer with the sources it "
        "grounded on: what Gemini tells people. Takes 10-90 s.",
    ),
    "copilot": (
        "Copilot",
        "Ask Microsoft Copilot Search on Bing and get its answer with cited sources and products. "
        "Takes 15-40 s.",
    ),
    "ai_mode": (
        "Google AI Mode",
        "Ask Google AI Mode and get its answer with cited sources and products. Takes 10-45 s.",
    ),
    "brave_ai": (
        "Brave AI",
        "Ask Brave Search's AI answer and get it with cited sources, the searches it ran and products. "
        "Takes about 10 s.",
    ),
}


def format_assistant(res: AssistantResponse) -> str:
    """The answer (Markdown with ``[n](link)`` citations), its numbered sources, products and
    the mentions of the tracked terms."""
    lines = [res.markdown or res.text or "The assistant gave no answer."]
    if res.sources:
        lines += ["", "Sources:"]
        lines += [
            f"[{s.position}] {s.title or s.domain} - {s.link}{'' if s.cited else ' (listed, not cited)'}"
            for s in res.sources
        ]
    if res.products:
        lines += ["", "Products:"]
        for p in res.products:
            extra = ", ".join(x for x in (p.price, p.merchant) if x)
            lines.append(f"{p.position}. {p.title}" + (f" ({extra})" if extra else ""))
    if res.mentions:
        lines += ["", "Mentions:"]
        for m in res.mentions:
            how = [w for w, on in (("named in the text", m.mentioned), ("cited", m.cited)) if on]
            if m.sources:
                how.append("sources " + ", ".join(str(n) for n in m.sources))
            if m.products:
                how.append("products " + ", ".join(str(n) for n in m.products))
            lines.append(f"- {m.term}: {'; '.join(how) or 'not mentioned'}")
    return "\n".join(lines)


class SerpKiteAssistant(BaseTool):
    """What an AI assistant answers, via SerpKite: ChatGPT (``/v1/chatgpt``), Gemini, Copilot
    Search, Google AI Mode or Brave AI, asked as a logged-out user.

    Returns the answer (Markdown with ``[n](link)`` citations) followed by its numbered sources,
    products and, with ``track``, how each brand or domain was mentioned. ``response_format`` is
    ``content_and_artifact``: the artifact is the full :class:`serpkite.types.AssistantResponse`
    (``sources``, ``search_queries``, ``products``, ``places``, ``mentions``). Prices per answer:
    ChatGPT 5, Gemini 5, Copilot 3, AI Mode 3, Brave AI 2 credits; failed and empty answers are
    free (https://serpkite.com/pricing).

    Instantiate:
        .. code-block:: python

            from langchain_serpkite import SerpKiteAssistant

            chatgpt = SerpKiteAssistant(assistant="chatgpt", track=["hubspot.com", "Pipedrive"])
            gemini = SerpKiteAssistant(assistant="gemini", country="de")

    Invoke:
        .. code-block:: python

            chatgpt.invoke({"prompt": "best crm for a 10-person startup"})
    """

    name: str = "serpkite_chatgpt"
    description: str = _ASSISTANT_INFO["chatgpt"][1]
    args_schema: type[BaseModel] = SerpKiteAssistantInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    api_wrapper: SerpKiteAPIWrapper = Field(default_factory=SerpKiteAPIWrapper)
    """Holds the API key and HTTP clients (``SERPKITE_API_KEY`` by default)."""
    assistant: Assistant = "chatgpt"
    """Which assistant to ask: chatgpt, gemini, copilot, ai_mode or brave_ai. Sets the default
    ``name`` (``serpkite_<assistant>``) and ``description``."""
    track: Optional[list[str]] = None
    """Brands (``"Acme"``) and domains (``"acme.com"``) to check each answer for (up to 20)."""
    country: Optional[str] = None
    """Ask from this country (e.g. ``de``). Defaults to ``api_wrapper.country``."""
    language: Optional[str] = None
    """Interface language (not ChatGPT). Defaults to ``api_wrapper.language``."""
    web_search: Optional[bool] = None
    """ChatGPT only: ``False`` lets ChatGPT decide whether to search (default: it searches)."""

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        values = _wrapper_from_kwargs(values)
        if isinstance(values, dict):
            assistant = values.get("assistant", "chatgpt")
            if assistant in _ASSISTANT_INFO:
                values.setdefault("name", f"serpkite_{assistant}")
                values.setdefault(
                    "description",
                    _ASSISTANT_INFO[assistant][1] + " Input is the prompt, as a user would type it.",
                )
        return values

    @model_validator(mode="after")
    def _check_options(self) -> SerpKiteAssistant:
        if self.web_search is not None and self.assistant != "chatgpt":
            raise ValueError("web_search is only supported by assistant='chatgpt'")
        if self.language is not None and self.assistant == "chatgpt":
            raise ValueError("language is not supported by assistant='chatgpt'")
        return self

    def _params(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "country": self.country,
            "language": self.language,
            "web_search": self.web_search,
        }

    def _run(
        self, prompt: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> tuple[str, AssistantResponse]:
        res = self.api_wrapper.assistant(self.assistant, prompt, **self._params())
        return format_assistant(res), res

    async def _arun(
        self, prompt: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> tuple[str, AssistantResponse]:
        res = await self.api_wrapper.aassistant(self.assistant, prompt, **self._params())
        return format_assistant(res), res
