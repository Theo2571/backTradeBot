"""
Core domain models for the DCA bot.

These are pure Python dataclasses / Pydantic models with no external
dependencies.  They represent the canonical state of the bot and its orders.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ======================================================================
# Enumerations
# ======================================================================


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"       # Created locally, not yet sent to Binance
    OPEN = "OPEN"             # Resting on the Binance order book
    FILLED = "FILLED"         # Fully executed
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class BotStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class CycleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


# ======================================================================
# Domain entities
# ======================================================================


class GridLevel(BaseModel):
    """
    One level in the DCA price grid.

    Attributes:
        index:          Zero-based position in the grid (0 = closest to market).
        price:          LIMIT order price for this level.
        quantity:       Base-asset quantity to buy at this level.
        quote_amount:   price * quantity (pre-calculated for validation).
    """

    index: int
    price: Decimal
    quantity: Decimal
    quote_amount: Decimal

    class Config:
        frozen = True


class Order(BaseModel):
    """
    Represents a single order tracked by the bot.

    Attributes:
        internal_id:       UUID assigned by the bot before sending to Binance.
        binance_order_id:  Binance-assigned numeric order ID (None until placed).
        client_order_id:   Custom client order ID sent to Binance for idempotency.
        symbol:            Trading pair (e.g., "BTCUSDT").
        side:              BUY or SELL.
        price:             Limit price.
        original_qty:      Quantity requested when the order was placed.
        executed_qty:      Quantity filled so far.
        commission:        Base-asset commission paid on fills.
        status:            Current order status.
        grid_index:        Grid level index (BUY orders only).
        created_at:        Timestamp when order was created locally.
        updated_at:        Timestamp of last status update.
    """

    internal_id: UUID = Field(default_factory=uuid4)
    binance_order_id: Optional[int] = None
    client_order_id: str = Field(default_factory=lambda: f"dca_{uuid4().hex[:16]}")
    symbol: str
    side: OrderSide
    price: Decimal
    original_qty: Decimal
    executed_qty: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    commission_asset: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    grid_index: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def net_qty(self) -> Decimal:
        """Quantity after subtracting commission (base-asset commissions only)."""
        return self.executed_qty - self.commission

    def model_copy_updated(self, **kwargs) -> "Order":
        """Return a new Order with updated fields and refreshed updated_at."""
        return self.model_copy(
            update={**kwargs, "updated_at": datetime.now(timezone.utc)}
        )


class DcaCycle(BaseModel):
    """
    Represents one complete DCA cycle from grid placement to sell execution.

    A cycle begins when grid BUY orders are placed and ends when the
    take-profit SELL order is fully filled (or the cycle is aborted).

    Attributes:
        cycle_id:          Unique identifier for this cycle.
        symbol:            Trading pair.
        status:            Current lifecycle stage.
        grid_levels:       Planned grid configuration for this cycle.
        buy_orders:        All BUY orders placed in this cycle.
        sell_order:        Current take-profit SELL order (may be replaced).
        first_buy_price:   Price of the topmost grid level (reference point).
        started_at:        When the cycle started.
        completed_at:      When the cycle ended (if completed/aborted).
    """

    cycle_id: UUID = Field(default_factory=uuid4)
    symbol: str
    status: CycleStatus = CycleStatus.ACTIVE
    grid_levels: list[GridLevel] = Field(default_factory=list)
    buy_orders: list[Order] = Field(default_factory=list)
    sell_order: Optional[Order] = None
    first_buy_price: Optional[Decimal] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def filled_buy_orders(self) -> list[Order]:
        """Return all BUY orders that have been fully filled."""
        return [o for o in self.buy_orders if o.is_filled]

    @property
    def active_buy_orders(self) -> list[Order]:
        """Return all BUY orders still resting on the order book."""
        return [o for o in self.buy_orders if o.is_active]

    @property
    def total_filled_qty(self) -> Decimal:
        """Total base-asset quantity acquired (net of commissions)."""
        return sum(
            (o.net_qty for o in self.filled_buy_orders),
            Decimal("0"),
        )

    @property
    def total_spent(self) -> Decimal:
        """Total quote-currency spent on filled BUY orders."""
        return sum(
            (o.price * o.executed_qty for o in self.filled_buy_orders),
            Decimal("0"),
        )

    @property
    def average_buy_price(self) -> Optional[Decimal]:
        """
        Volume-weighted average price of all filled BUY orders.

        Returns None if no orders have been filled yet.
        """
        if not self.filled_buy_orders:
            return None
        total_qty = sum(o.executed_qty for o in self.filled_buy_orders)
        if total_qty == 0:
            return None
        return self.total_spent / total_qty
