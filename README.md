# langchain-serpkite

LangChain integration for [SerpKite](https://serpkite.com), the Google search API built for AI
agents. The package gives you:

| Class | What it does |
| --- | --- |
| `SerpKiteSearch` | Tool (`serpkite_search`) that returns Markdown, which uses few tokens, for agents |
| `SerpKiteSearchResults` | Tool (`serpkite_search_results_json`) that returns a list of result dicts (title, link, domain, snippet, …) |
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

By default every request is answered by Google only (`engine="google"`); SerpKite already fails
over across its own proxy pools. Opt in to other providers with `engine` on a tool, the retriever
or the shared wrapper (you set it, never the model):

```python
search = SerpKiteSearch(engine="auto")                          # fall back when Google is unavailable
retriever = SerpKiteRetriever(k=5, engine=["google", "brave"])  # only these providers, in order
docs = retriever.invoke("espresso", engine="auto")              # per-call override
docs[0].metadata["engine"]                                      # "google", or e.g. "brave"
```

- `engine` is `"google"` (default), `"auto"`, `"consensus"`, one provider (`"brave"`, `"bing"`,
  `"yahoo"`, `"duckduckgo"`, `"mojeek"`, `"wikipedia"`) or a list. `"auto"` and `"consensus"` can't be combined
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
