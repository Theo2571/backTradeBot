"""
Pydantic models for API request/response serialisation.

Contract is driven by the frontend (frontTradeBot) types:
  - All responses wrapped in ApiResponse[T]
  - Request fields are camelCase (matching TypeScript BotConfig)
  - Status response matches TypeScript BotStatus interface
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ======================================================================
# Generic API wrapper  —  matches frontend ApiResponse<T>
# ======================================================================

class ApiResponse(BaseModel, Generic[T]):
    """
    Envelope used for every response.

    Frontend expects: { success: bool, data: T, message?: string, error?: string }
    """

    success: bool
    data: T
    message: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: T, message: Optional[str] = None) -> "ApiResponse[T]":
        return cls(success=True, data=data, message=message)

    @classmethod
    def fail(cls, error: str, data: T = None) -> "ApiResponse[T]":  # type: ignore[assignment]
        return cls(success=False, data=data, error=error)


# ======================================================================
# Request model  —  matches frontend BotConfig
# ======================================================================

class StartBotRequest(BaseModel):
    """
    Payload sent by the frontend when starting the bot.

    Field names match the TypeScript BotConfig interface exactly (camelCase).
    """

    # Credentials (accepted per-request so the user doesn't need a .env)
    apiKey: str = Field(..., min_length=10, description="Binance API key")
    apiSecret: str = Field(..., min_length=10, description="Binance API secret")

    # Trading pair
    pair: str = Field(
        ...,
        description="Trading pair, e.g. BTCUSDT",
        examples=["BTCUSDT"],
    )

    # Capital
    depositAmount: float = Field(..., gt=0, description="Quote-currency capital per cycle")

    # Grid shape
    gridRangePercent: float = Field(..., gt=0, lt=100, description="Total price span of the grid (%)")
    offsetPercent: float = Field(..., ge=0, lt=50, description="% below market price for first BUY")
    ordersCount: int = Field(..., ge=2, le=100, description="Number of grid BUY levels")

    # Volumes
    # Frontend sends as a growth percentage (e.g. 50 = each next order 50% larger → multiplier 1.5).
    # 0 means equal volume for every level (multiplier 1.0).
    volumeScalePercent: float = Field(..., ge=0, le=400, description="Volume growth per level (%, 0 = equal volumes)")

    # Take-profit and grid shift
    takeProfitPercent: float = Field(..., gt=0, lt=100, description="Take-profit % above avg buy price")
    gridShiftPercent: float = Field(..., ge=0, le=50, description="Market rise % that triggers grid shift (upward only)")

    # Environment
    isTestnet: bool = Field(default=True, description="True = Binance testnet")


class VerifyCredentialsRequest(BaseModel):
    """POST /api/verify-credentials — same keys as start, but no trading params."""

    apiKey: str = Field(..., min_length=10, description="Binance API key")
    apiSecret: str = Field(..., min_length=10, description="Binance API secret")
    isTestnet: bool = Field(default=True, description="True = Binance testnet")


# ======================================================================
# Response data shapes  —  match frontend TypeScript interfaces
# ======================================================================

class OrderData(BaseModel):
    """Matches frontend Order interface."""

    id: str
    price: float
    amount: float
    total: float
    status: str  # 'active' | 'executed' | 'cancelled'
    executedAt: Optional[str] = None
    profit: Optional[float] = None


class ActiveSellOrder(BaseModel):
    """Matches frontend activeSellOrder shape."""

    price: float
    amount: float


class BotStatusData(BaseModel):
    """
    Matches frontend BotStatus interface exactly.

    status:               'running' | 'stopped' | 'error'
    currentPrice:         Last known market price
    executedOrdersCount:  Number of filled BUY orders in current cycle
    averagePrice:         Volume-weighted average of filled BUY prices
    activeSellOrder:      Current take-profit SELL (null if none)
    totalProfit:          Cumulative realised profit across all cycles
    activeOrders:         Open BUY orders in current grid
    executedOrders:       Filled BUY orders in current cycle
    """

    status: str
    currentPrice: float
    executedOrdersCount: int
    averagePrice: float
    activeSellOrder: Optional[ActiveSellOrder] = None
    totalProfit: float
    lastCycleProfit: Optional[float] = None
    currentPnl: float = 0.0
    cyclesCompleted: int = 0
    activeOrders: list[OrderData]
    executedOrders: list[OrderData]


class StartBotData(BaseModel):
    """Data payload inside the start response."""
    message: str


class StopBotData(BaseModel):
    """Data payload inside the stop response."""
    message: str


class AssetBalance(BaseModel):
    """Single asset balance."""
    asset: str
    free: float
    locked: float


class VerifyCredentialsData(BaseModel):
    """Summary returned after a successful Binance account read (signed GET /api/v3/account)."""

    accountType: str
    canTrade: bool
    canWithdraw: bool
    canDeposit: bool
    permissions: list[str]
    nonZeroBalances: int
    totalBalanceEntries: int


class BalanceData(BaseModel):
    """
    Account balances for the current trading pair assets.

    `balances` keeps the full rows for the UI list.
    Explicit quote/base fields are the single source for limits (e.g. deposit ≤ quoteFree).
    """

    balances: list[AssetBalance]
    pair: str
    quoteAsset: str = Field(..., description="Quote asset of the pair, e.g. USDT")
    quoteFree: float = Field(..., description="Free balance available for new BUY margin")
    quoteLocked: float = Field(..., description="Quote locked in open orders")
    baseAsset: str = Field(..., description="Base asset of the pair, e.g. BTC")
    baseFree: float = Field(..., description="Free base balance (sellable)")
    baseLocked: float = Field(..., description="Base locked in open orders")
    usdtFree: Optional[float] = Field(
        None,
        description="Same as quoteFree when quote is USDT; otherwise null",
    )
    updatedAt: str = Field(..., description="ISO 8601 UTC timestamp when balances were read")
