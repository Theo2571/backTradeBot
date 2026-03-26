"""
FastAPI router — API layer.

Endpoints match what the frontend (frontTradeBot) calls:
  POST /api/start   → start the DCA bot
  POST /api/stop    → stop the bot gracefully
  GET  /api/status  → current bot state snapshot

All responses are wrapped in ApiResponse[T]:
  { success: bool, data: T, message?: str, error?: str }

Dynamic Binance credentials are accepted in the start request body so
the user doesn't need a server-side .env for their API keys.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from infrastructure.binance_client import BinanceClient, BinanceClientError, InsufficientBalanceError
from models.api_models import (
    ActiveSellOrder,
    ApiResponse,
    AssetBalance,
    BalanceData,
    BotStatusData,
    OrderData,
    StartBotData,
    StartBotRequest,
    StopBotData,
    VerifyCredentialsData,
    VerifyCredentialsRequest,
)
from domain.models import BotStatus as EngineBotStatus
from services.bot_service import BotService

logger = get_logger(__name__)

_ALLOWED_PREVIEW_PAIRS = frozenset({"BTCUSDT", "ETHUSDT"})
router = APIRouter(tags=["DCA Bot"])


def get_bot_service(request: Request) -> BotService:
    """Dependency: retrieve BotService from app state."""
    return request.app.state.bot_service


BotServiceDep = Annotated[BotService, Depends(get_bot_service)]


# ======================================================================
# POST /verify-credentials
# ======================================================================

@router.post(
    "/verify-credentials",
    summary="Verify Binance API keys (read-only account check)",
)
async def verify_credentials(
    bot: BotServiceDep,
    body: VerifyCredentialsRequest,
) -> ApiResponse[VerifyCredentialsData]:
    """
    Validate that the provided API key/secret can sign requests against the
    chosen environment (testnet or mainnet). Does not start the bot.

    On success, the same credentials are applied to the shared BotService client
    so GET /balance and /status use the user's keys (not placeholder .env keys).
    """
    logger.info("api_verify_credentials_request", testnet=body.isTestnet)

    client = BinanceClient(
        api_key=body.apiKey.strip(),
        api_secret=body.apiSecret.strip(),
        testnet=body.isTestnet,
        dry_run=False,
        max_retries=2,
        retry_delay=1.0,
    )

    try:
        account = await client._call(client._client.get_account)
    except BinanceClientError as exc:
        logger.warning("verify_credentials_failed", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=ApiResponse.fail(str(exc)).model_dump(),
        )
    except Exception as exc:
        logger.exception("verify_credentials_unexpected", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse.fail("Verification failed — check server logs").model_dump(),
        )

    balances_raw: list[dict] = account.get("balances", [])
    non_zero = 0
    for b in balances_raw:
        try:
            free = float(b.get("free", 0))
            locked = float(b.get("locked", 0))
        except (TypeError, ValueError):
            continue
        if free > 0 or locked > 0:
            non_zero += 1

    perms = account.get("permissions")
    if not isinstance(perms, list):
        perms = []

    data = VerifyCredentialsData(
        accountType=str(account.get("accountType", "UNKNOWN")),
        canTrade=bool(account.get("canTrade", False)),
        canWithdraw=bool(account.get("canWithdraw", False)),
        canDeposit=bool(account.get("canDeposit", False)),
        permissions=[str(p) for p in perms],
        nonZeroBalances=non_zero,
        totalBalanceEntries=len(balances_raw),
    )

    bot.apply_exchange_credentials(
        body.apiKey.strip(),
        body.apiSecret.strip(),
        body.isTestnet,
    )
    await bot.refresh_spot_price_if_idle()

    return ApiResponse.ok(data=data, message="Credentials valid")


# ======================================================================
# POST /start
# ======================================================================

@router.post(
    "/start",
    response_model=ApiResponse[StartBotData],
    summary="Start the DCA bot",
)
async def start_bot(
    bot: BotServiceDep,
    body: StartBotRequest,
) -> ApiResponse[StartBotData]:
    """
    Start the DCA bot.

    Credentials and all trading parameters are supplied in the request body.
    The bot runs in the background and is monitored via GET /status.
    """
    logger.info(
        "api_start_request",
        pair=body.pair,
        deposit=body.depositAmount,
        levels=body.ordersCount,
        testnet=body.isTestnet,
        dry_run=bot._settings.dry_run,
    )

    # Apply request parameters to settings + recreate client with fresh creds
    _apply_request_to_bot(bot, body)

    try:
        await bot.start()
    except RuntimeError as exc:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=ApiResponse.fail(str(exc)).model_dump(),
        )
    except (ValueError, InsufficientBalanceError) as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ApiResponse.fail(str(exc)).model_dump(),
        )
    except BinanceClientError as exc:
        logger.error("start_binance_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ApiResponse.fail(f"Binance API error: {exc}").model_dump(),
        )
    except Exception as exc:
        logger.exception("start_unexpected_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse.fail("Unexpected error — check server logs").model_dump(),
        )

    return ApiResponse.ok(
        data=StartBotData(message="Bot started successfully"),
        message="DCA bot is now running",
    )


# ======================================================================
# POST /stop
# ======================================================================

@router.post(
    "/stop",
    response_model=ApiResponse[StopBotData],
    summary="Stop the DCA bot",
)
async def stop_bot(bot: BotServiceDep) -> ApiResponse[StopBotData]:
    """Stop the bot and cancel all open orders on Binance."""
    logger.info("api_stop_request")
    try:
        await bot.stop()
    except Exception as exc:
        logger.exception("stop_unexpected_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse.fail("Error during stop — check server logs").model_dump(),
        )

    return ApiResponse.ok(
        data=StopBotData(message="Bot stopped and all open orders cancelled"),
        message="Stopped",
    )


# ======================================================================
# GET /status
# ======================================================================

@router.get(
    "/status",
    response_model=ApiResponse[BotStatusData],
    summary="Get bot status",
)
async def get_status(
    bot: BotServiceDep,
    pair: Annotated[
        Optional[str],
        Query(description="Spot pair for market price preview when bot is idle (BTCUSDT | ETHUSDT)"),
    ] = None,
) -> ApiResponse[BotStatusData]:
    """
    Return the current state of the bot and active cycle.

    Polled by the frontend every 2.5 s when the bot is running.
    """
    if bot._status in (EngineBotStatus.RUNNING, EngineBotStatus.STOPPING):
        await bot.refresh_spot_price_if_idle(None)
    else:
        pq = pair.strip().upper() if pair and pair.strip() else None
        if pq and pq not in _ALLOWED_PREVIEW_PAIRS:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ApiResponse.fail("pair must be BTCUSDT or ETHUSDT").model_dump(),
            )
        await bot.refresh_spot_price_if_idle(pq)

    raw = bot.get_status()

    active_sell = None
    if raw.get("activeSellOrder"):
        active_sell = ActiveSellOrder(
            price=raw["activeSellOrder"]["price"],
            amount=raw["activeSellOrder"]["amount"],
        )

    data = BotStatusData(
        status=raw["status"],
        currentPrice=raw["currentPrice"],
        executedOrdersCount=raw["executedOrdersCount"],
        averagePrice=raw["averagePrice"],
        activeSellOrder=active_sell,
        totalProfit=raw["totalProfit"],
        lastCycleProfit=raw.get("lastCycleProfit"),
        currentPnl=raw.get("currentPnl", 0.0),
        cyclesCompleted=raw.get("cyclesCompleted", 0),
        activeOrders=[OrderData(**o) for o in raw["activeOrders"]],
        executedOrders=[OrderData(**o) for o in raw["executedOrders"]],
    )
    return ApiResponse.ok(data=data)


# ======================================================================
# GET /balance
# ======================================================================

@router.get(
    "/balance",
    response_model=ApiResponse[BalanceData],
    summary="Get account balances for current pair assets",
)
async def get_balance(
    bot: BotServiceDep,
    pair: Annotated[
        Optional[str],
        Query(description="Spot pair for balance preview (BTCUSDT | ETHUSDT); ignored while bot runs"),
    ] = None,
) -> ApiResponse[BalanceData]:
    """
    Return free + locked balances for both assets of the current trading pair.
    Credentials must be applied (e.g. after /verify-credentials).
    """
    if bot._status in (EngineBotStatus.RUNNING, EngineBotStatus.STOPPING):
        symbol = bot._settings.symbol
    elif pair and pair.strip():
        sym = pair.strip().upper()
        if sym not in _ALLOWED_PREVIEW_PAIRS:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ApiResponse.fail("pair must be BTCUSDT or ETHUSDT").model_dump(),
            )
        symbol = sym
    else:
        symbol = bot._settings.symbol

    try:
        # Use get_symbol_info to get exact base/quote asset names from Binance
        symbol_info = await bot._client.get_symbol_info(symbol)
        base_asset = symbol_info.base_asset
        quote_asset = symbol_info.quote_asset
        target_assets = {base_asset, quote_asset}

        # Fetch full account — _call wraps asyncio.to_thread with retry logic
        account = await bot._client._call(bot._client._client.get_account)
        balances_raw: list[dict] = account.get("balances", [])

        by_asset: dict[str, dict] = {b["asset"]: b for b in balances_raw}

        def _fl(asset: str) -> tuple[float, float]:
            row = by_asset.get(asset)
            if not row:
                return 0.0, 0.0
            return float(row["free"]), float(row["locked"])

        quote_free, quote_locked = _fl(quote_asset)
        base_free, base_locked = _fl(base_asset)

        result: list[AssetBalance] = [
            AssetBalance(asset=base_asset, free=base_free, locked=base_locked),
            AssetBalance(asset=quote_asset, free=quote_free, locked=quote_locked),
        ]
        result.sort(key=lambda x: x.asset)

        usdt_free: float | None = quote_free if quote_asset == "USDT" else None
        updated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        return ApiResponse.ok(
            data=BalanceData(
                balances=result,
                pair=symbol,
                quoteAsset=quote_asset,
                quoteFree=quote_free,
                quoteLocked=quote_locked,
                baseAsset=base_asset,
                baseFree=base_free,
                baseLocked=base_locked,
                usdtFree=usdt_free,
                updatedAt=updated,
            )
        )

    except BinanceClientError as exc:
        logger.error("balance_binance_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ApiResponse.fail(f"Binance API error: {exc}").model_dump(),
        )
    except Exception as exc:
        logger.exception("balance_unexpected_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse.fail("Could not fetch balance").model_dump(),
        )


# ======================================================================
# Internal helpers
# ======================================================================

def _apply_request_to_bot(bot: BotService, req: StartBotRequest) -> None:
    """
    Push all request fields into the bot's settings and rebuild the client.

    volumeScalePercent is sent by the frontend as a percentage (e.g. 150),
    but GridConfig expects a plain multiplier (e.g. 1.5), so we divide by 100.
    """
    s = bot._settings

    # Trading parameters
    s.__dict__["symbol"] = req.pair.upper()
    s.__dict__["deposit_amount"] = Decimal(str(req.depositAmount))
    s.__dict__["grid_levels"] = req.ordersCount  # validated >= 2 by api_models
    s.__dict__["initial_offset_percent"] = Decimal(str(req.offsetPercent))
    s.__dict__["grid_range_percent"] = Decimal(str(req.gridRangePercent))
    # Frontend sends 150 meaning 1.5×; domain layer uses the multiplier directly
    # Frontend sends volumeScalePercent as a growth % (e.g. 50 = 50% growth → multiplier 1.5).
    # Domain GridConfig expects a plain multiplier (e.g. 1.5), so we convert:
    #   multiplier = 1 + growth_percent / 100
    # Special case: 0% means equal volumes for all levels (multiplier = 1.0).
    s.__dict__["volume_scale_percent"] = Decimal(str(1 + req.volumeScalePercent / 100))
    s.__dict__["profit_percent"] = Decimal(str(req.takeProfitPercent))
    s.__dict__["grid_shift_threshold_percent"] = Decimal(str(req.gridShiftPercent))

    # Environment + Binance client (same as /verify-credentials)
    bot.apply_exchange_credentials(req.apiKey, req.apiSecret, req.isTestnet)
