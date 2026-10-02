"""SerpKiteWebpageLoader: any public URL as a clean-Markdown Document."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from serpkite import SerpKiteError
from serpkite.types import WebpageResponse

from ._utilities import SerpKiteAPIWrapper

__all__ = ["SerpKiteWebpageLoader"]

logger = logging.getLogger(__name__)


class SerpKiteWebpageLoader(BaseLoader):
    """Load web pages as Markdown Documents through SerpKite ``/v1/webpage`` (1 credit per page).

    Setup:
        .. code-block:: bash

            pip install -U langchain-serpkite
            export SERPKITE_API_KEY="skt_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_serpkite import SerpKiteWebpageLoader

            loader = SerpKiteWebpageLoader(["https://example.com", "https://serpkite.com"])

    Load:
        .. code-block:: python

            docs = loader.load()
            docs[0].page_content  # Markdown
            docs[0].metadata      # {"source": ..., "title": ..., "description": ..., ...}

    Async:
        .. code-block:: python

            docs = [doc async for doc in loader.alazy_load()]
    """

    def __init__(
        self,
        urls: Union[str, Sequence[str]],
        *,
        include_html: bool = False,
        continue_on_failure: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_wrapper: Optional[SerpKiteAPIWrapper] = None,
    ) -> None:
        """
        Args:
            urls: One URL or a list of URLs to load.
            include_html: Also put the raw HTML into ``metadata["html"]``.
            continue_on_failure: Log and skip URLs that fail instead of raising.
            api_key: SerpKite API key (defaults to ``SERPKITE_API_KEY``).
            base_url: API origin override.
            api_wrapper: A preconfigured :class:`SerpKiteAPIWrapper` (overrides api_key/base_url).
        """
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.include_html = include_html
        self.continue_on_failure = continue_on_failure
        if api_wrapper is None:
            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            api_wrapper = SerpKiteAPIWrapper(**kwargs)
        self.api_wrapper = api_wrapper

    def _to_document(self, requested: str, page: WebpageResponse) -> Document:
        metadata: dict[str, Any] = {"source": page.url or requested, "requested_url": requested}
        if page.status_code is not None:
            metadata["status_code"] = page.status_code
        for key, value in page.metadata.model_dump(exclude_none=True).items():
            if value not in ("", None):
                metadata[key] = value
        if self.include_html and page.html:
            metadata["html"] = page.html
        return Document(page_content=page.markdown, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        for url in self.urls:
            try:
                page = self.api_wrapper.webpage(url, include_html=self.include_html)
            except SerpKiteError as exc:
                if not self.continue_on_failure:
                    raise
                logger.warning("SerpKite could not load %s: %s", url, exc)
                continue
            yield self._to_document(url, page)

    async def alazy_load(self) -> AsyncIterator[Document]:
        for url in self.urls:
            try:
                page = await self.api_wrapper.awebpage(url, include_html=self.include_html)
            except SerpKiteError as exc:
                if not self.continue_on_failure:
                    raise
                logger.warning("SerpKite could not load %s: %s", url, exc)
                continue
            yield self._to_document(url, page)
