"""Web search adapter. The graph depends only on `SearchFn`, so the provider is swappable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SearchFn = Callable[[str], list[dict[str, str]]]


def make_tavily_search(max_results: int = 3, client: Any = None) -> SearchFn:
    if client is None:
        from tavily import TavilyClient

        client = TavilyClient()  # reads TAVILY_API_KEY

    def search(query: str) -> list[dict[str, str]]:
        response = client.search(query=query, max_results=max_results, search_depth="advanced")
        return [
            {"title": r["title"], "content": r["content"], "url": r["url"]}
            for r in response.get("results", [])
        ]

    return search
