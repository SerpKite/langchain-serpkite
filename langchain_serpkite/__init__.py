"""LangChain integration for SerpKite, the Google search API built for AI agents."""

from importlib import metadata

from langchain_serpkite._utilities import SerpKiteAPIWrapper
from langchain_serpkite.document_loaders import SerpKiteWebpageLoader
from langchain_serpkite.retrievers import SerpKiteRetriever
from langchain_serpkite.tools import (
    SerpKiteAnswer,
    SerpKiteAnswerInput,
    SerpKiteAssistant,
    SerpKiteAssistantInput,
    SerpKiteResearch,
    SerpKiteResearchInput,
    SerpKiteSearch,
    SerpKiteSearchInput,
    SerpKiteSearchResults,
)

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.4.0"
del metadata

__all__ = [
    "SerpKiteAPIWrapper",
    "SerpKiteAnswer",
    "SerpKiteAnswerInput",
    "SerpKiteAssistant",
    "SerpKiteAssistantInput",
    "SerpKiteResearch",
    "SerpKiteResearchInput",
    "SerpKiteRetriever",
    "SerpKiteSearch",
    "SerpKiteSearchInput",
    "SerpKiteSearchResults",
    "SerpKiteWebpageLoader",
    "__version__",
]
