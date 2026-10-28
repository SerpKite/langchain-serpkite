# langchain-serpkite

LangChain integration for [SerpKite](https://serpkite.com), the Google search API built for AI
agents. The package gives you:

| Class | What it does |
| --- | --- |
| `SerpKiteSearch` | Tool (`serpkite_search`) that returns Markdown, which uses few tokens, for agents |
| `SerpKiteSearchResults` | Tool (`serpkite_search_results_json`) that returns a list of result dicts (title, link, domain, snippet, …) |
| `SerpKiteAnswer` | Tool (`serpkite_answer`) that returns a cited answer from live Google results, with the numbered sources |
| `SerpKiteResearch` | Tool (`serpkite_research`) that runs a deep-research task and returns its cited Markdown report |
| `SerpKiteAssistant` | Tool (`serpkite_chatgpt`, `serpkite_gemini`, …) that asks ChatGPT, Gemini, Copilot, Google AI Mode or Brave AI and returns its answer, sources, products and brand `mentions` |
| `SerpKiteRetriever` | Retriever that returns Google results as `Document`s, optionally with full page Markdown |
| `SerpKiteWebpageLoader` | Document loader that fetches any public URL as clean Markdown |
| `SerpKiteAPIWrapper` | Holds the API key and the sync and async SerpKite clients |

Every class supports both sync and async (`invoke`/`ainvoke`, `load`/`alazy_load`).

## Install

```bash
pip install -U langchain-serpkite
export SERPKITE_API_KEY=skt_live_...   # https://app.serpkite.com/keys
```

## Tools

```python
from langchain_serpkite import SerpKiteSearch, SerpKiteSearchResults

search = SerpKiteSearch()  # optional: endpoint="news", country="de", num=20, api_key="..."
print(search.invoke({"query": "best espresso machine"}))  # Markdown string

papers = SerpKiteSearch(endpoint="scholar")  # or news, images, videos, maps, places,
                                             # shopping, patents, autocomplete
rows = SerpKiteSearchResults(max_results=5).invoke({"query": "langgraph checkpointer"})
# [{"position": 1, "title": "...", "link": "https://...", "domain": "...", "snippet": "..."}, ...]
```

The model can set `query`, `num`, `country`, `language` and `time`
(`hour|day|week|month|year`). Defaults set on the tool (`country=`, `language=`, `location=`,
`num=`) apply when the model leaves a field out. `engine=` is set on the tool only (see
[Search engines & fallback](#search-engines--fallback)).

## Cited answers and research

```python
from langchain_serpkite import SerpKiteAnswer, SerpKiteResearch

answer = SerpKiteAnswer(depth="fast")  # "standard" (default) also reads the top num_sources pages
print(answer.invoke({"query": "what is the latest stable rust version"}))
# The latest stable Rust release is … [1].
#
# Sources:
# [1] Rust blog - https://blog.rust-lang.org/...

research = SerpKiteResearch(effort="lite")  # lite | standard (default) | deep
print(research.invoke({"query": "compare managed Postgres vector search options"}))
```

- Both tools use `response_format="content_and_artifact"`: called with a tool call, the
  `ToolMessage.artifact` is the full `serpkite.types.AnswerResponse` / `ResearchTask` (citations,
  sources, `meta.credits`, `meta.llm`).
- `SerpKiteAnswer` takes `depth`, `num_sources` (1-5), `include_domains`, `exclude_domains`,
  `country`, `language` and `time` on the tool; the model only sets `query`.
- `SerpKiteResearch` starts the task and waits for the report (one to several minutes, up to
  `timeout=900` seconds). If the wait runs out it returns the task's state; the task keeps running
  server-side. `effort` sets how much is searched and read.
- When the model fails, `SerpKiteAnswer` raises `serpkite.APIError` with code `ai_unavailable`
  and `SerpKiteResearch` returns the failed task's error; neither is charged for the model part.
  Prices: [serpkite.com/pricing](https://serpkite.com/pricing).

## AI assistants: ChatGPT, Gemini, Copilot, AI Mode, Brave AI

```python
from langchain_serpkite import SerpKiteAssistant

chatgpt = SerpKiteAssistant(assistant="chatgpt", track=["hubspot.com", "Pipedrive"])
print(chatgpt.invoke({"prompt": "best crm for a 10-person startup"}))
# HubSpot … [1](https://hubspot.com/…)
#
# Sources:
# [1] HubSpot - https://hubspot.com/…
#
# Mentions:
# - hubspot.com: named in the text; cited; sources 1
# - Pipedrive: not mentioned

gemini = SerpKiteAssistant(assistant="gemini", country="de", language="de")  # serpkite_gemini
tools = [SerpKiteAssistant(assistant=a) for a in ("chatgpt", "gemini", "copilot", "ai_mode", "brave_ai")]
```

- `assistant` is `chatgpt` (default), `gemini`, `copilot`, `ai_mode` or `brave_ai`; it sets the
  tool's default `name` (`serpkite_<assistant>`) and description. The model only sets `prompt`.
- On the tool: `track` (brands and domains to check each answer for, up to 20), `country`,
  `language` (not ChatGPT) and `web_search` (ChatGPT only; `False` lets ChatGPT decide).
- The content is the answer as Markdown with `[n](link)` citations, then its sources, products
  and how each `track` term was mentioned. The `ToolMessage.artifact` is the full
  `serpkite.types.AssistantResponse` (`sources`, `search_queries`, `products`, `places`,
  `mentions`).
- Answers take 10-90 s. Prices per answer: ChatGPT 4, Gemini 3, Copilot 3, AI Mode 2, Brave AI 2
  credits; failed and empty answers are free.

## Agent (LangChain / LangGraph)

```python
from langchain.agents import create_agent
from langchain_serpkite import SerpKiteSearch, SerpKiteSearchResults

agent = create_agent(
    model="provider:model-name",  # any tool-calling chat model
    tools=[SerpKiteSearch(), SerpKiteSearchResults(max_results=5)],
    system_prompt="You are a research assistant. Search before answering and cite links.",
)
out = agent.invoke({"messages": [{"role": "user", "content": "What changed in the EU AI Act this month?"}]})
print(out["messages"][-1].content)
```

`create_agent` runs on LangGraph. The tools work the same way with
`langgraph.prebuilt.create_react_agent(model, tools=[...])`, and in a custom `StateGraph` through
`ToolNode`.

## Retriever

```python
from langchain_serpkite import SerpKiteRetriever

retriever = SerpKiteRetriever(k=5, include_content=2, country="us", language="en")
docs = retriever.invoke("how does HNSW indexing work")
docs = retriever.invoke("how does HNSW indexing work", k=3)  # per-call override

docs[0].page_content  # page Markdown for the top include_content results, else the snippet
docs[0].metadata      # {"title", "link", "source", "position", "domain", "snippet", "engine", ...}
```

- `include_content` (0-5) fetches the top N result pages as Markdown for +1 credit each.
- `k` above 10 uses a deep search, which costs 7 credits for 100 results.
- `time` limits results to the last hour, day, week, month or year.

A minimal RAG chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("Answer from these sources:\n{context}\n\nQuestion: {question}")
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

## Webpage loader

```python
from langchain_serpkite import SerpKiteWebpageLoader

loader = SerpKiteWebpageLoader(["https://example.com", "https://serpkite.com"], continue_on_failure=True)
docs = loader.load()                               # sync
docs = [d async for d in loader.alazy_load()]      # async
docs[0].metadata  # {"source", "requested_url", "status_code", "title", "description", "language", ...}
```

Each page costs 1 credit. Pass `include_html=True` to also get the raw HTML in
`metadata["html"]`.

## Search engines & fallback

By default every request is answered by Google only (`engine="google"`); SerpKite already
retries Google internally. Opt in to other providers with `engine` on a tool, the retriever
or the shared wrapper (you set it, never the model):

```python
search = SerpKiteSearch(engine="auto")                          # fall back when Google is unavailable
retriever = SerpKiteRetriever(k=5, engine=["google", "brave"])  # only these providers, in order
docs = retriever.invoke("espresso", engine="auto")              # per-call override
docs[0].metadata["engine"]                                      # "google", or e.g. "brave"
```

- `engine` is `"google"` (default), `"auto"`, `"consensus"`, one provider (`"brave"`, `"bing"`,
  `"yahoo"`, `"duckduckgo"`, `"mojeek"`, `"wikipedia"`, `"yandex"`) or a list. `"auto"` and `"consensus"` can't be combined
  with other names; unknown names, or a provider that doesn't serve the endpoint, raise
  `serpkite.BadRequestError`.
- The retriever puts the answering provider (`meta.engine`) into each Document's `metadata["engine"]`.
  With `engine="consensus"` (several indexes merged and ranked by agreement; costs the sum of the
  providers that answered) it is `"consensus"`, and `metadata["sources"]` lists the providers that
  returned each result.
- Credits follow the answering provider's price.

## Configuration

Every class takes `api_key=` and `base_url=`, or a shared wrapper:

```python
from langchain_serpkite import SerpKiteAPIWrapper, SerpKiteRetriever, SerpKiteSearch

wrapper = SerpKiteAPIWrapper(api_key="skt_live_...", country="de", language="de", max_retries=3)
tool = SerpKiteSearch(api_wrapper=wrapper)
retriever = SerpKiteRetriever(api_wrapper=wrapper, k=5)
```

API errors raise `serpkite.SerpKiteError` subclasses. Each one has `status`, `code`, `message` and
`request_id`. See the [`serpkite`](https://pypi.org/project/serpkite/) SDK. Failed, empty and
blocked searches are not billed.

## Development

```bash
uv sync                      # installs ../python (serpkite) in editable mode
uv run pytest                # unit tests: mocked HTTP, sockets disabled, LangChain standard tests
uv run ruff check . && uv run ruff format --check .
uv run mypy
SERPKITE_API_KEY=skt_live_... uv run pytest tests/integration_tests   # live
```

## License

MIT
