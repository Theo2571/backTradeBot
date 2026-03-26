"""Tests for POST /api/verify-credentials."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.router import router


@pytest.fixture
def mock_bot_service() -> MagicMock:
    bot = MagicMock()
    bot.apply_exchange_credentials = MagicMock()
    bot.refresh_spot_price_if_idle = AsyncMock()
    return bot


@pytest.fixture
def verify_app(mock_bot_service: MagicMock) -> FastAPI:
    app = FastAPI()
    app.state.bot_service = mock_bot_service
    app.include_router(router, prefix="/api")
    return app


@pytest.mark.asyncio
async def test_verify_credentials_success(
    verify_app: FastAPI, mock_bot_service: MagicMock
):
    fake_account = {
        "accountType": "SPOT",
        "canTrade": True,
        "canWithdraw": True,
        "canDeposit": True,
        "permissions": ["SPOT"],
        "balances": [
            {"asset": "BTC", "free": "0.1", "locked": "0"},
            {"asset": "USDT", "free": "0", "locked": "0"},
        ],
    }

    with patch("api.router.BinanceClient") as MockClient:
        instance = MagicMock()
        instance._client = MagicMock()
        instance._call = AsyncMock(return_value=fake_account)
        MockClient.return_value = instance

        async with AsyncClient(
            transport=ASGITransport(app=verify_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/verify-credentials",
                json={
                    "apiKey": "a" * 32,
                    "apiSecret": "b" * 32,
                    "isTestnet": True,
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["accountType"] == "SPOT"
    assert body["data"]["canTrade"] is True
    assert body["data"]["nonZeroBalances"] == 1
    assert body["data"]["totalBalanceEntries"] == 2
    mock_bot_service.apply_exchange_credentials.assert_called_once()
    mock_bot_service.refresh_spot_price_if_idle.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_credentials_invalid(
    verify_app: FastAPI, mock_bot_service: MagicMock
):
    from infrastructure.binance_client import BinanceClientError

    with patch("api.router.BinanceClient") as MockClient:
        instance = MagicMock()
        instance._client = MagicMock()

        async def fail_call(*_a, **_kw):
            raise BinanceClientError("Invalid API-key, IP, or permissions for action.")

        instance._call = fail_call
        MockClient.return_value = instance

        async with AsyncClient(
            transport=ASGITransport(app=verify_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/verify-credentials",
                json={
                    "apiKey": "a" * 32,
                    "apiSecret": "b" * 32,
                    "isTestnet": True,
                },
            )

    assert resp.status_code == 401
    assert resp.json()["success"] is False
    assert "Invalid" in resp.json()["error"]
    mock_bot_service.apply_exchange_credentials.assert_not_called()
