from __future__ import annotations

from typing import Any

BASE = "https://api.serpkite.com"


def meta() -> dict[str, Any]:
    return {"request_id": "req_1", "credits_used": 1, "cached": False}


def organic(n: int, *, content_for: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        row: dict[str, Any] = {
            "position": i,
            "title": f"Result {i}",
            "link": f"https://site{i}.example/page",
            "domain": f"site{i}.example",
            "snippet": f"Snippet {i}",
        }
        if i <= content_for:
            row["content"] = f"# Page {i}\n\nFull text {i}"
        rows.append(row)
    return rows


def search_body(n: int = 10, *, content_for: int = 0) -> dict[str, Any]:
    return {
        "request": {"endpoint": "search", "engine": "google"},
        "results": organic(n, content_for=content_for),
        "related_searches": [],
        "meta": meta(),
    }


def webpage_body(url: str) -> dict[str, Any]:
    return {
        "request": {"endpoint": "webpage", "engine": "http", "url": url},
        "url": url + "/",
        "status_code": 200,
        "markdown": "# Example Domain\n\nThis domain is for examples.",
        "text": "Example Domain This domain is for examples.",
        "metadata": {"title": "Example Domain", "description": "", "language": "en"},
        "meta": meta(),
    }
