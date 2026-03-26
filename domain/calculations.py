"""
Pure mathematical functions for DCA grid construction and order sizing.

All calculations use Decimal arithmetic to avoid floating-point rounding
errors that would be unacceptable in a real-money trading context.

This module has NO side effects and NO external dependencies — it is
purely functional and fully unit-testable in isolation.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import NamedTuple

from .models import GridLevel


class GridConfig(NamedTuple):
    """Input parameters for grid calculation."""

    market_price: Decimal
    initial_offset_percent: Decimal  # e.g. Decimal("1.0") = 1%
    grid_range_percent: Decimal       # e.g. Decimal("5.0") = 5%
    grid_levels: int
    deposit_amount: Decimal
    volume_scale_percent: Decimal     # multiplier per level, e.g. Decimal("1.5")
    price_precision: int = 2          # decimal places for price
    qty_precision: int = 6            # decimal places for quantity


class TakeProfitParams(NamedTuple):
    """Input for take-profit price calculation."""

    average_price: Decimal
    profit_percent: Decimal  # e.g. Decimal("2.0") = 2%
    price_precision: int = 2


def calculate_grid_prices(config: GridConfig) -> list[Decimal]:
    """
    Calculate the limit-buy prices for each grid level.

    Level 0  (closest to market):  P1 = market * (1 - offset%)
    Level N-1 (furthest):          P_last = P1 * (1 - range%)

    Steps are distributed linearly between P1 and P_last.

    Args:
        config: Grid configuration parameters.

    Returns:
        List of Decimal prices, index 0 being the highest (nearest market).

    Raises:
        ValueError: If parameters produce an invalid grid (e.g. P_last >= P1).

    Example:
        >>> cfg = GridConfig(
        ...     market_price=Decimal("100"),
        ...     initial_offset_percent=Decimal("1"),
        ...     grid_range_percent=Decimal("5"),
        ...     grid_levels=5,
        ...     deposit_amount=Decimal("1000"),
        ...     volume_scale_percent=Decimal("1.0"),
        ... )
        >>> prices = calculate_grid_prices(cfg)
        >>> len(prices)
        5
    """
    if config.grid_levels < 2:
        raise ValueError("grid_levels must be at least 2")

    HUNDRED = Decimal("100")
    p1 = config.market_price * (1 - config.initial_offset_percent / HUNDRED)
    p_last = p1 * (1 - config.grid_range_percent / HUNDRED)

    if p_last >= p1:
        raise ValueError(
            f"Invalid grid: p_last ({p_last}) must be less than p1 ({p1}). "
            "Increase grid_range_percent."
        )
    if p_last <= 0:
        raise ValueError("Grid produces non-positive prices. Reduce grid_range_percent.")

    step = (p1 - p_last) / (config.grid_levels - 1)
    quantize_price = Decimal(10) ** -config.price_precision

    return [
        (p1 - step * i).quantize(quantize_price, rounding=ROUND_DOWN)
        for i in range(config.grid_levels)
    ]


def calculate_grid_volumes(config: GridConfig) -> list[Decimal]:
    """
    Calculate base-asset quantities for each grid level.

    Per spec (point 7): the USDT allocation for each level grows geometrically —
      usdt_i = usdt_0 * scale^i
    so that the *quote-currency spend* per order increases by a fixed multiplier,
    matching the user's intent of "second order costs n% more than the first".

    Base-asset quantity is then derived as:
      qty_i = usdt_i / price_i

    The USDT allocations are normalised so their sum equals exactly deposit_amount
    (rounding down each quantity means the actual total may be slightly below).

    Args:
        config: Grid configuration parameters.

    Returns:
        List of Decimal quantities (base asset), one per grid level.

    Raises:
        ValueError: If deposit is not positive or weights sum to zero.
    """
    if config.deposit_amount <= 0:
        raise ValueError("deposit_amount must be positive")

    prices = calculate_grid_prices(config)
    scale = config.volume_scale_percent  # plain multiplier, e.g. 1.5 for 50% growth

    # USDT weight for each level — grows geometrically by scale
    usdt_weights = [scale**i for i in range(config.grid_levels)]
    total_weight = sum(usdt_weights)

    if total_weight == 0:
        raise ValueError("USDT weight sum is zero — check volume_scale_percent.")

    quantize_qty = Decimal(10) ** -config.qty_precision

    # usdt_i = deposit * weight_i / total_weight  →  qty_i = usdt_i / price_i
    quantities = [
        (config.deposit_amount * usdt_weights[i] / total_weight / prices[i]).quantize(
            quantize_qty, rounding=ROUND_DOWN
        )
        for i in range(config.grid_levels)
    ]

    return quantities


def build_grid(config: GridConfig) -> list[GridLevel]:
    """
    Construct the full DCA grid as a list of GridLevel objects.

    Args:
        config: Grid configuration parameters.

    Returns:
        Ordered list of GridLevel, index 0 = closest to market price.

    Raises:
        ValueError: If the grid cannot be constructed with given parameters.
        ValueError: If the total grid cost would exceed deposit_amount.
    """
    prices = calculate_grid_prices(config)
    quantities = calculate_grid_volumes(config)

    levels: list[GridLevel] = []
    total_cost = Decimal("0")

    for i in range(config.grid_levels):
        quote_amount = prices[i] * quantities[i]
        total_cost += quote_amount
        levels.append(
            GridLevel(
                index=i,
                price=prices[i],
                quantity=quantities[i],
                quote_amount=quote_amount,
            )
        )

    # Sanity check: total cost must not exceed deposit
    if total_cost > config.deposit_amount:
        raise ValueError(
            f"Grid total cost {total_cost} exceeds deposit {config.deposit_amount}. "
            "This can happen due to rounding; try reducing grid_levels or qty_precision."
        )

    return levels


def calculate_average_price(filled_orders: list[tuple[Decimal, Decimal]]) -> Decimal:
    """
    Calculate the volume-weighted average price of filled orders.

    Args:
        filled_orders: List of (price, executed_qty) tuples.

    Returns:
        VWAP as a Decimal.

    Raises:
        ValueError: If the list is empty or total quantity is zero.

    Example:
        >>> calculate_average_price([
        ...     (Decimal("100"), Decimal("0.5")),
        ...     (Decimal("95"),  Decimal("1.0")),
        ... ])
        Decimal('96.666666...')
    """
    if not filled_orders:
        raise ValueError("No filled orders to average.")

    total_cost = sum(price * qty for price, qty in filled_orders)
    total_qty = sum(qty for _, qty in filled_orders)

    if total_qty == 0:
        raise ValueError("Total executed quantity is zero.")

    return total_cost / total_qty


def calculate_net_quantity(
    filled_orders: list[tuple[Decimal, Decimal]],
) -> Decimal:
    """
    Calculate total base-asset quantity net of commissions.

    Args:
        filled_orders: List of (executed_qty, commission) tuples.
                       commission is expected to be in base asset.

    Returns:
        Net Decimal quantity available for the SELL order.
    """
    total_qty = sum(qty for qty, _ in filled_orders)
    total_commission = sum(commission for _, commission in filled_orders)
    return total_qty - total_commission


def calculate_take_profit_price(params: TakeProfitParams) -> Decimal:
    """
    Calculate the SELL limit price for the take-profit order.

    sell_price = avg_price * (1 + profit% / 100)

    Args:
        params: Average price and profit target.

    Returns:
        Quantised take-profit price.

    Raises:
        ValueError: If average_price is not positive.
    """
    if params.average_price <= 0:
        raise ValueError("average_price must be positive")
    if params.profit_percent <= 0:
        raise ValueError("profit_percent must be positive")

    HUNDRED = Decimal("100")
    raw = params.average_price * (1 + params.profit_percent / HUNDRED)
    quantize_price = Decimal(10) ** -params.price_precision
    return raw.quantize(quantize_price, rounding=ROUND_DOWN)


def should_shift_grid(
    first_buy_price: Decimal,
    current_market_price: Decimal,
    threshold_percent: Decimal,
) -> bool:
    """
    Determine whether the price has risen far enough to warrant a grid shift.

    A shift is warranted when the market has moved UP past the first (highest)
    BUY level by more than threshold_percent, meaning our grid is now too low
    to be filled in a reasonable timeframe.

    Args:
        first_buy_price:    Price of the topmost grid BUY level.
        current_market_price: Current Binance spot mid-price.
        threshold_percent:  Minimum upward move (%) that triggers a shift.

    Returns:
        True if the grid should be cancelled and recalculated.
    """
    if threshold_percent <= 0:
        return False  # 0 means "grid shift disabled"

    if first_buy_price <= 0 or current_market_price <= 0:
        return False

    # How far above first_buy_price is the market?
    deviation = (current_market_price - first_buy_price) / first_buy_price * 100
    return deviation > threshold_percent
