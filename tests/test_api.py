"""
Integration tests for the FastAPI endpoints.

Uses httpx AsyncClient + ASGITransport so no real server is started.
BotService is replaced with a MagicMock — no Binance connection.

Response format: ApiResponse[T] = { success, data, message?, error? }
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from config.settings import Environment, Settings
from domain.models import BotStatus
from infrastructure.binance_client import BinanceClientError, InsufficientBalanceError
from services.bot_service import BotService


# ======================================================================
# Shared test body — valid StartBotRequest payload
# ======================================================================

VALID_START_BODY = {
    "apiKey": "test_api_key_1234",
    "apiSecret": "test_api_secret_1234",
    "pair": "BTCUSDT",
    "depositAmount": 1000.0,
    "gridRangePercent": 5.0,
    "offsetPercent": 1.0,
    "ordersCount": 5,
    "volumeScalePercent": 50.0,
    "takeProfitPercent": 2.0,
    "gridShiftPercent": 1.0,
    "isTestnet": True,
}

# Minimal get_status() return that the router can map to BotStatusData
IDLE_STATUS = {
    "status": "stopped",
    "currentPrice": 0.0,
    "executedOrdersCount": 0,
    "averagePrice": 0.0,
    "activeSellOrder": None,
    "totalProfit": 0.0,
    "lastCycleProfit": None,
    "currentPnl": 0.0,
    "cyclesCompleted": 0,
    "activeOrders": [],
    "executedOrders": [],
}

RUNNING_STATUS = {
    **IDLE_STATUS,
    "status": "running",
    "currentPrice": 50000.0,
    "executedOrdersCount": 2,
    "averagePrice": 49500.0,
    "cyclesCompleted": 1,
    "activeOrders": [
        {
            "id": "order-1",
            "price": 49000.0,
            "amount": 0.001,
            "total": 49.0,
            "status": "active",
            "executedAt": None,
            "profit": None,
        }
    ],
    "executedOrders": [
        {
            "id": "order-2",
            "price": 50000.0,
            "amount": 0.001,
            "total": 50.0,
            "status": "executed",
            "executedAt": "2024-01-01T00:00:00+00:00",
            "profit": None,
        }
    ],
}


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        binance_api_key="test_key",
        binance_api_secret="test_secret",
        environment=Environment.TESTNET,
        dry_run=True,
        symbol="BTCUSDT",
        deposit_amount=Decimal("1000"),
        grid_levels=5,
        initial_offset_percent=Decimal("1"),
        grid_range_percent=Decimal("5"),
        volume_scale_percent=Decimal("1.5"),
        profit_percent=Decimal("2"),
        grid_shift_threshold_percent=Decimal("1"),
        poll_interval_seconds=5,
        api_max_retries=3,
        api_retry_delay_seconds=2,
    )


@pytest.fixture
def mock_bot_service(mock_settings: Settings) -> MagicMock:
    service = MagicMock(spec=BotService)
    service._settings = mock_settings
    service._status = BotStatus.IDLE
    service.start = AsyncMock()
    service.stop = AsyncMock()
    service.apply_exchange_credentials = MagicMock()
    service.refresh_spot_price_if_idle = AsyncMock()
    service.get_status = MagicMock(return_value=IDLE_STATUS)
    return service


@pytest.fixture
def test_app(mock_bot_service: MagicMock) -> FastAPI:
    """FastAPI app with mocked BotService injected directly into app state."""
    from api.router import router

    app = FastAPI()
    app.include_router(router, prefix="/bot")
    # Inject directly — no lifespan needed in tests
    app.state.bot_service = mock_bot_service
    return app


@pytest_asyncio.fixture
async def client(test_app: FastAPI) -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as c:
        yield c


# ======================================================================
# POST /bot/start
# ======================================================================

@pytest.mark.asyncio
async def test_start_returns_200_on_success(
    client: AsyncClient, mock_bot_service: MagicMock
):
    resp = await client.post("/bot/start", json=VALID_START_BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "message" in body["data"]
    mock_bot_service.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_already_running_returns_409(
    client: AsyncClient, mock_bot_service: MagicMock
):
    mock_bot_service.start.side_effect = RuntimeError("Bot is already running")
    resp = await client.post("/bot/start", json=VALID_START_BODY)
    assert resp.status_code == 409
    body = resp.json()
    assert body["success"] is False
    assert "already running" in body["error"]


@pytest.mark.asyncio
async def test_start_insufficient_balance_returns_400(
    client: AsyncClient, mock_bot_service: MagicMock
):
    mock_bot_service.start.side_effect = InsufficientBalanceError("Not enough USDT")
    resp = await client.post("/bot/start", json=VALID_START_BODY)
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "Not enough USDT" in body["error"]


@pytest.mark.asyncio
async def test_start_invalid_config_returns_400(
    client: AsyncClient, mock_bot_service: MagicMock
):
    mock_bot_service.start.side_effect = ValueError("Invalid grid configuration")
    resp = await client.post("/bot/start", json=VALID_START_BODY)
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


@pytest.mark.asyncio
async def test_start_binance_error_returns_502(
    client: AsyncClient, mock_bot_service: MagicMock
):
    mock_bot_service.start.side_effect = BinanceClientError("Exchange unreachable")
    resp = await client.post("/bot/start", json=VALID_START_BODY)
    assert resp.status_code == 502
    body = resp.json()
    assert body["success"] is False
    assert "Binance API error" in body["error"]


@pytest.mark.asyncio
async def test_start_missing_body_returns_422(client: AsyncClient):
    """FastAPI schema validation must reject an empty request body."""
    resp = await client.post("/bot/start")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_start_invalid_api_key_too_short_returns_422(client: AsyncClient):
    """apiKey with fewer than 10 characters must fail Pydantic validation."""
    body = {**VALID_START_BODY, "apiKey": "short"}
    resp = await client.post("/bot/start", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_start_invalid_orders_count_returns_422(client: AsyncClient):
    """ordersCount < 2 must be rejected by schema validation."""
    body = {**VALID_START_BODY, "ordersCount": 1}
    resp = await client.post("/bot/start", json=body)
    assert resp.status_code == 422


# ======================================================================
# POST /bot/stop
# ======================================================================

@pytest.mark.asyncio
async def test_stop_running_bot_returns_200(
    client: AsyncClient, mock_bot_service: MagicMock
):
    resp = await client.post("/bot/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "stopped" in body["data"]["message"].lower()
    mock_bot_service.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_when_already_idle_is_safe(
    client: AsyncClient, mock_bot_service: MagicMock
):
    """Calling stop on an idle bot must not raise — graceful noop."""
    resp = await client.post("/bot/stop")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ======================================================================
# GET /bot/status
# ======================================================================

@pytest.mark.asyncio
async def test_get_status_idle(client: AsyncClient, mock_bot_service: MagicMock):
    resp = await client.get("/bot/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["status"] == "stopped"
    assert data["currentPrice"] == 0.0
    assert data["activeOrders"] == []
    assert data["executedOrders"] == []


@pytest.mark.asyncio
async def test_get_status_running_with_active_orders(
    client: AsyncClient, mock_bot_service: MagicMock
):
    mock_bot_service.get_status.return_value = RUNNING_STATUS
    resp = await client.get("/bot/status")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "running"
    assert data["currentPrice"] == 50000.0
    assert data["executedOrdersCount"] == 2
    assert data["averagePrice"] == 49500.0
    assert len(data["activeOrders"]) == 1
    assert len(data["executedOrders"]) == 1


@pytest.mark.asyncio
async def test_get_status_response_envelope_shape(
    client: AsyncClient, mock_bot_service: MagicMock
):
    """Every response must be wrapped in ApiResponse — success + data at minimum."""
    resp = await client.get("/bot/status")
    body = resp.json()
    assert "success" in body
    assert "data" in body
    # FastAPI serialises missing Optional fields as null, not omitting them
    assert "error" in body


@pytest.mark.asyncio
async def test_get_status_with_invalid_pair_returns_400(
    client: AsyncClient, mock_bot_service: MagicMock
):
    """Unrecognised pair query param must be rejected."""
    resp = await client.get("/bot/status", params={"pair": "DOGEUSDT"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_status_refreshes_price_when_idle(
    client: AsyncClient, mock_bot_service: MagicMock
):
    """GET /status must call refresh_spot_price_if_idle regardless of bot state."""
    await client.get("/bot/status")
    mock_bot_service.refresh_spot_price_if_idle.assert_awaited_once()
