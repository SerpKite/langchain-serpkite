from __future__ import annotations

import pytest
import serpkite._client as client_mod


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPKITE_API_KEY", "skt_live_test")
    monkeypatch.delenv("SERPKITE_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_async_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(client_mod, "_sleep", lambda seconds: None)
    monkeypatch.setattr(client_mod, "_async_sleep", fake_async_sleep)
