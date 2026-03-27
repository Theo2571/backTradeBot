"""
BotService — orchestration layer.

This is the core state machine of the DCA bot. It owns the lifecycle of
a DCA cycle:

  IDLE ─── start() ──► RUNNING ─── (cycle fills + sell) ──► IDLE
                                  └─── stop() ──────────────► IDLE

Threading model:
  - The FastAPI endpoints call start/stop/status from the async event loop.
  - The bot's main loop runs as a background asyncio.Task.
  - All Binance I/O is awaited (runs in a thread pool under the hood).

State ownership:
  - BotService owns a single DcaCycle at a time.
  - State is in-memory only (no DB).  For persistence, inject a repository.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Optional

from config.logging_config import get_logger
from config.settings import Environment, Settings
from domain.calculations import (
    GridConfig,
    TakeProfitParams,
    build_grid,
    calculate_average_price,
    calculate_net_quantity,
    calculate_take_profit_price,
    should_shift_grid,
)
from domain.models import (
    BotStatus,
    CycleStatus,
    DcaCycle,
    Order,
    OrderSide,
    OrderStatus,
)
from infrastructure.binance_client import BinanceClient, BinanceClientError, InsufficientBalanceError

logger = get_logger(__name__)


class BotService:
    """
    Orchestrates the full DCA cycle for a single trading pair.

    Responsibilities:
    - Validate configuration and account balance before starting.
    - Build and place grid BUY orders.
    - Monitor order fills (polling loop).
    - Place / replace take-profit SELL orders as BUYs fill.
    - Detect grid shift condition and re-anchor the grid.
    - Restart automatically when a cycle completes.
    - Cancel all orders cleanly on stop().

    Args:
        settings:   Application configuration (injected for testability).
        client:     Binance API wrapper (injected for testability / dry-run).
    """

    def __init__(self, settings: Settings, client: BinanceClient) -> None:
        self._settings = settings
        self._client = client
        self._status = BotStatus.IDLE
        self._cycle: Optional[DcaCycle] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._cycles_completed = 0
        self._error_message: Optional[str] = None
        self._last_price = Decimal("0")
        self._total_profit = Decimal("0")
        self._last_cycle_profit: Optional[Decimal] = None
        self._completed_sell_orders: list[dict] = []

    def apply_exchange_credentials(
        self,
        api_key: str,
        api_secret: str,
        is_testnet: bool,
    ) -> None:
        """
        Install API keys on the shared Binance client.

        Called after POST /verify-credentials (and /start does a full apply including symbol).
        Without this, GET /balance would still use placeholder keys from server startup.

        If the bot is already RUNNING or STOPPING, only the client is swapped — the in-memory
        cycle, background task, and stop_event are preserved so a UI re-login does not wipe
        active orders and state.
        """
        s = self._settings
        env = Environment.TESTNET if is_testnet else Environment.MAINNET
        s.__dict__["environment"] = env
        self._client = BinanceClient(
            api_key=api_key.strip(),
            api_secret=api_secret.strip(),
            testnet=is_testnet,
            dry_run=s.dry_run,
            max_retries=s.api_max_retries,
            retry_delay=s.api_retry_delay_seconds,
        )

        current = getattr(self, "_status", BotStatus.IDLE)
        if current in (BotStatus.RUNNING, BotStatus.STOPPING):
            logger.info(
                "apply_exchange_credentials_hot_swap",
                status=current.value,
                message="Preserving bot state; only API client updated",
            )
            return

        self._status = BotStatus.IDLE
        self._cycle: Optional[DcaCycle] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._cycles_completed: int = 0
        self._error_message: Optional[str] = None
        self._last_price: Decimal = Decimal("0")
        self._total_profit: Decimal = Decimal("0")
        self._last_cycle_profit: Optional[Decimal] = None
        self._completed_sell_orders: list[dict] = []

    async def refresh_spot_price_if_idle(self, symbol_override: Optional[str] = None) -> None:
        """
        Fetch spot price from Binance while the bot is not running.

        ``currentPrice`` in GET /status comes from ``_last_price``, which is only
        updated inside the trading loop unless we refresh here — otherwise it
        stays 0 after login (apply_exchange_credentials resets it).

        ``symbol_override`` lets the UI request a price for a pair selected in the
        form before ``start`` (must be a supported spot symbol).
        """
        if self._status not in (BotStatus.IDLE, BotStatus.ERROR):
            return
        allowed = {"BTCUSDT", "ETHUSDT"}
        sym = (symbol_override or "").strip().upper() if symbol_override else self._settings.symbol
        if sym not in allowed:
            sym = self._settings.symbol
        try:
            self._last_price = await self._client.get_symbol_price(sym)
        except BinanceClientError:
            pass

    # ------------------------------------------------------------------ #
    # Public interface (called from API layer)
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Start the bot.

        Validates configuration and spawns the background loop.

        Raises:
            RuntimeError: If the bot is already running.
            ValueError:   If configuration is invalid.
            InsufficientBalanceError: If account balance is too low.
        """
        if self._status == BotStatus.RUNNING:
            raise RuntimeError("Bot is already running")
        if self._status == BotStatus.STOPPING:
            raise RuntimeError("Bot is currently stopping — wait before restarting")

        await self._validate_pre_start()

        self._stop_event.clear()
        self._error_message = None
        self._status = BotStatus.RUNNING

        self._task = asyncio.create_task(
            self._run_loop(),
            name="dca_bot_loop",
        )
        logger.info(
            "bot_started",
            symbol=self._settings.symbol,
            dry_run=self._settings.dry_run,
            environment=self._settings.environment.value,
        )

    async def stop(self) -> None:
        """
        Request a graceful shutdown.

        Signals the run loop to stop after the current iteration,
        then cancels all open orders and waits for the task to finish.
        """
        if self._status not in (BotStatus.RUNNING, BotStatus.ERROR):
            return

        logger.info("bot_stop_requested")
        self._status = BotStatus.STOPPING
        self._stop_event.set()

        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("bot_task_timeout_canceling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        await self._cancel_all_orders()
        self._status = BotStatus.IDLE
        logger.info("bot_stopped")

    def get_status(self) -> dict:
        """
        Return a structured snapshot matching the frontend BotStatus interface.

        {
          status: 'running' | 'stopped' | 'error'
          currentPrice: float
          executedOrdersCount: int
          averagePrice: float
          activeSellOrder: { price, amount } | null
          totalProfit: float
          activeOrders: [Order]
          executedOrders: [Order]
        }
        """
        # Map internal BotStatus → frontend strings
        status_map = {
            BotStatus.IDLE: "stopped",
            BotStatus.RUNNING: "running",
            BotStatus.STOPPING: "running",
            BotStatus.ERROR: "error",
        }
        frontend_status = status_map.get(self._status, "stopped")

        # Build order lists from current cycle
        active_orders: list[dict] = []
        executed_orders: list[dict] = []
        avg_price = 0.0
        executed_count = 0
        active_sell: Optional[dict] = None

        if self._cycle:
            avg = self._cycle.average_buy_price
            avg_price = float(avg) if avg else 0.0
            executed_count = len(self._cycle.filled_buy_orders)

            # Active (open) BUY orders
            for o in self._cycle.active_buy_orders:
                active_orders.append({
                    "id": str(o.binance_order_id or o.internal_id),
                    "price": float(o.price),
                    "amount": float(o.original_qty),
                    "total": float(o.price * o.original_qty),
                    "status": "active",
                    "executedAt": None,
                    "profit": None,
                })

            # Filled BUY orders
            for o in self._cycle.filled_buy_orders:
                executed_orders.append({
                    "id": str(o.binance_order_id or o.internal_id),
                    "price": float(o.price),
                    "amount": float(o.executed_qty),
                    "total": float(o.price * o.executed_qty),
                    "status": "executed",
                    "executedAt": o.updated_at.isoformat(),
                    "profit": None,
                })

            # Active SELL order
            sell = self._cycle.sell_order
            if sell and sell.is_active:
                active_sell = {
                    "price": float(sell.price),
                    "amount": float(sell.original_qty),
                }

        # Unrealized P&L for the current open cycle (zero if no buys yet)
        current_pnl = 0.0
        if self._cycle and self._cycle.total_filled_qty > 0 and self._last_price > 0:
            net_qty = self._cycle.total_filled_qty
            total_spent = self._cycle.total_spent
            current_pnl = float(self._last_price * net_qty - total_spent)

        return {
            "status": frontend_status,
            "currentPrice": float(self._last_price),
            "executedOrdersCount": executed_count,
            "averagePrice": avg_price,
            "activeSellOrder": active_sell,
            "totalProfit": float(self._total_profit),
            "lastCycleProfit": float(self._last_cycle_profit) if self._last_cycle_profit is not None else None,
            "currentPnl": current_pnl,
            "cyclesCompleted": self._cycles_completed,
            "activeOrders": active_orders,
            "executedOrders": self._completed_sell_orders + executed_orders,
        }

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    async def _run_loop(self) -> None:
        """
        Background loop that drives the DCA state machine.

        Each iteration:
        1. Start a new cycle (place grid orders).
        2. Poll order statuses until stop is requested.
        3. On SELL fill → mark complete, restart.
        4. On stop request → clean up.
        """
        try:
            while not self._stop_event.is_set():
                await self._run_single_cycle()

                if self._stop_event.is_set():
                    break

                self._cycles_completed += 1
                logger.info(
                    "cycle_completed",
                    total_cycles=self._cycles_completed,
                )
                # Brief pause before starting next cycle
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            logger.info("bot_loop_cancelled")
        except Exception as exc:
            self._status = BotStatus.ERROR
            self._error_message = str(exc)
            logger.exception("bot_loop_unhandled_error", error=str(exc))
            await self._cancel_all_orders()

    async def _run_single_cycle(self) -> None:
        """
        Execute one full DCA cycle.

        Raises on unrecoverable errors; returns normally on success.
        """
        logger.info("starting_new_cycle", symbol=self._settings.symbol)

        # 1. Get current market price
        market_price = await self._client.get_symbol_price(self._settings.symbol)
        self._last_price = market_price
        logger.info("market_price_fetched", price=str(market_price))

        # 2. Get symbol precision info
        sym_info = await self._client.get_symbol_info(self._settings.symbol)

        # 2a. Balance check before every cycle (not just on manual Start).
        #     Stops the bot cleanly if the account no longer has enough funds
        #     instead of silently placing only part of the grid.
        if not self._settings.dry_run:
            balance = await self._client.get_account_balance(sym_info.quote_asset)
            if balance < self._settings.deposit_amount:
                raise InsufficientBalanceError(
                    f"Insufficient {sym_info.quote_asset} balance to start a new cycle. "
                    f"Have: {balance:.2f}, Need: {self._settings.deposit_amount:.2f}. "
                    "Top up your account or reduce the deposit amount."
                )

        # 3. Build grid
        grid_config = GridConfig(
            market_price=market_price,
            initial_offset_percent=self._settings.initial_offset_percent,
            grid_range_percent=self._settings.grid_range_percent,
            grid_levels=self._settings.grid_levels,
            deposit_amount=self._settings.deposit_amount,
            volume_scale_percent=self._settings.volume_scale_percent,
            price_precision=sym_info.price_precision,
            qty_precision=sym_info.qty_precision,
        )
        grid = build_grid(grid_config)

        logger.info(
            "grid_calculated",
            levels=len(grid),
            top_price=str(grid[0].price),
            bottom_price=str(grid[-1].price),
            total_cost=str(sum(l.quote_amount for l in grid)),
        )

        # 4. Create new cycle
        self._cycle = DcaCycle(
            symbol=self._settings.symbol,
            grid_levels=grid,
            first_buy_price=grid[0].price,
        )

        # 5. Place all BUY orders
        await self._place_buy_orders()

        # 6. Polling loop
        await self._monitor_cycle()

    async def _place_buy_orders(self) -> None:
        """Place all grid BUY orders for the current cycle."""
        assert self._cycle is not None
        orders: list[Order] = []

        for i, level in enumerate(self._cycle.grid_levels):
            # Small pause between placements to avoid API rate-limit burst.
            # Binance allows 1200 weight/min; each POST /order = 1 weight.
            # 50 ms gap keeps 20-order grids well within limits.
            if i > 0:
                await asyncio.sleep(0.05)
            order = Order(
                symbol=self._cycle.symbol,
                side=OrderSide.BUY,
                price=level.price,
                original_qty=level.quantity,
                grid_index=level.index,
            )
            try:
                response = await self._client.place_limit_buy(
                    symbol=order.symbol,
                    price=order.price,
                    quantity=order.original_qty,
                    client_order_id=order.client_order_id,
                )
                order = order.model_copy_updated(
                    binance_order_id=response["orderId"],
                    status=OrderStatus.OPEN,
                )
                logger.info(
                    "buy_order_placed",
                    grid_index=level.index,
                    price=str(level.price),
                    qty=str(level.quantity),
                    order_id=response["orderId"],
                )
            except BinanceClientError as exc:
                logger.error(
                    "buy_order_failed",
                    grid_index=level.index,
                    error=str(exc),
                )
                order = order.model_copy_updated(status=OrderStatus.REJECTED)

            orders.append(order)

        self._cycle.buy_orders = orders

    async def _monitor_cycle(self) -> None:
        """
        Poll Binance until the cycle completes, is aborted, or stop is requested.
        """
        assert self._cycle is not None

        while not self._stop_event.is_set():
            await asyncio.sleep(self._settings.poll_interval_seconds)

            if self._stop_event.is_set():
                break

            try:
                # Refresh cached price on every tick
                try:
                    self._last_price = await self._client.get_symbol_price(
                        self._settings.symbol
                    )
                except BinanceClientError:
                    pass  # keep stale price — non-fatal

                await self._refresh_buy_orders()
                await self._check_sell_order()
                await self._check_grid_shift()
            except BinanceClientError as exc:
                logger.error("monitor_loop_error", error=str(exc))
                continue

            # Cycle completed when sell fills
            if self._cycle.status == CycleStatus.COMPLETED:
                # Track realised profit
                self._accrue_profit()
                logger.info(
                    "sell_filled_cycle_complete",
                    avg_price=str(self._cycle.average_buy_price),
                    sell_price=str(
                        self._cycle.sell_order.price if self._cycle.sell_order else "N/A"
                    ),
                    total_qty=str(self._cycle.total_filled_qty),
                    total_profit=str(self._total_profit),
                )
                return

            # Cycle aborted (e.g., after grid shift failure)
            if self._cycle.status == CycleStatus.ABORTED:
                logger.warning("cycle_aborted")
                return

    # ------------------------------------------------------------------ #
    # Order lifecycle helpers
    # ------------------------------------------------------------------ #

    async def _refresh_buy_orders(self) -> None:
        """
        Poll Binance for fresh status on all active BUY orders.

        Updates local order state and, when new fills are detected,
        triggers take-profit order placement/replacement.
        """
        assert self._cycle is not None
        newly_filled: list[Order] = []

        for i, order in enumerate(self._cycle.buy_orders):
            if not order.is_active:
                continue
            if order.binance_order_id is None:
                continue

            raw = await self._client.get_order_status(
                order.symbol, order.binance_order_id
            )
            updated = self._parse_order_response(order, raw)
            self._cycle.buy_orders[i] = updated

            if updated.is_filled and not order.is_filled:
                # Fetch real commission from trade history — GET /api/v3/order
                # does NOT return fills for resting limit orders, only the
                # initial POST response does.  myTrades is the only reliable source.
                # Pass base_asset so we only count commission paid in BTC (not BNB).
                sym_info = await self._client.get_symbol_info(updated.symbol)
                commission = await self._fetch_order_commission(
                    updated.symbol, updated.binance_order_id, sym_info.base_asset
                )
                if commission > Decimal("0"):
                    updated = updated.model_copy_updated(commission=commission)
                    self._cycle.buy_orders[i] = updated  # persist commission into cycle state

                logger.info(
                    "buy_order_filled",
                    grid_index=order.grid_index,
                    price=str(updated.price),
                    qty=str(updated.executed_qty),
                    commission=str(updated.commission),
                )
                newly_filled.append(updated)

        if newly_filled:
            await self._update_sell_order()

    async def _check_sell_order(self) -> None:
        """Check if the active SELL order has been filled."""
        assert self._cycle is not None

        sell = self._cycle.sell_order

        # If sell is missing or rejected but we have filled buys → retry placement.
        # This recovers the bot from a transient sell-placement failure without
        # requiring a restart (which would lose all in-memory state).
        if self._cycle.filled_buy_orders:
            sell_needs_retry = sell is None or sell.status in (
                OrderStatus.REJECTED,
                OrderStatus.CANCELED,
                OrderStatus.EXPIRED,
            )
            if sell_needs_retry:
                logger.warning(
                    "sell_order_missing_or_failed_retrying",
                    sell_status=sell.status.value if sell else None,
                    filled_count=len(self._cycle.filled_buy_orders),
                )
                await self._update_sell_order()
                return

        if sell is None or not sell.is_active:
            return
        if sell.binance_order_id is None:
            return

        raw = await self._client.get_order_status(sell.symbol, sell.binance_order_id)
        updated = self._parse_order_response(sell, raw)

        if updated.is_filled:
            # Fetch sell-side commission (paid in quote asset, e.g. USDT).
            # Must be fetched before _accrue_profit reads sell.commission.
            sell_commission = await self._fetch_order_commission(
                updated.symbol, updated.binance_order_id
            )
            if sell_commission > Decimal("0"):
                updated = updated.model_copy_updated(commission=sell_commission)
            self._cycle.status = CycleStatus.COMPLETED
            self._cycle.completed_at = datetime.now(timezone.utc)

        self._cycle.sell_order = updated

    def _accrue_profit(self) -> None:
        """
        Add realised profit from the completed cycle to the running total.

        revenue        = sell_price × net_qty   (net_qty = filled_qty minus buy commissions)
        profit         = revenue − total_spent − sell_commission
        total_spent    = Σ(price_i × executed_qty_i) for all filled BUY orders
        sell_commission = quote-asset fee deducted from sale proceeds
        """
        if self._cycle is None:
            return
        sell = self._cycle.sell_order
        net_qty = self._cycle.total_filled_qty
        total_spent = self._cycle.total_spent
        if sell is None or net_qty <= 0 or total_spent <= 0:
            return
        revenue = sell.price * net_qty
        profit = revenue - total_spent - sell.commission
        self._last_cycle_profit = profit
        self._total_profit += profit
        self._completed_sell_orders.append({
            "id": str(sell.binance_order_id or sell.internal_id),
            "price": float(sell.price),
            "amount": float(net_qty),
            "total": float(sell.price * net_qty),
            "status": "executed",
            "executedAt": sell.updated_at.isoformat(),
            "profit": float(profit),
        })
        logger.info(
            "profit_accrued",
            cycle_profit=str(profit),
            revenue=str(revenue),
            total_spent=str(total_spent),
            sell_commission=str(sell.commission),
            total_profit=str(self._total_profit),
        )

    async def _update_sell_order(self) -> None:
        """
        Place or replace the take-profit SELL order.

        Called whenever a new BUY fills.  The existing SELL (if any) is
        cancelled before the new one is placed to avoid double-sells.
        """
        assert self._cycle is not None

        filled = self._cycle.filled_buy_orders
        if not filled:
            return

        # Recalculate average price
        avg_price = calculate_average_price(
            [(o.price, o.executed_qty) for o in filled]
        )

        # Recalculate net quantity (subtract commissions)
        net_qty = calculate_net_quantity(
            [(o.executed_qty, o.commission) for o in filled]
        )

        if net_qty <= 0:
            logger.warning("net_qty_zero_skip_sell", net_qty=str(net_qty))
            return

        sym_info = await self._client.get_symbol_info(self._cycle.symbol)

        # Quantize net_qty to exchange LOT_SIZE step — raw qty after commission
        # subtraction may have more decimal places than Binance allows (e.g.
        # 0.00031968 vs step_size 0.00001), causing -1013 filter failure.
        quantize_qty = Decimal(10) ** -sym_info.qty_precision
        net_qty = net_qty.quantize(quantize_qty, rounding=ROUND_DOWN)

        if net_qty <= 0:
            logger.warning("net_qty_zero_after_quantize", net_qty=str(net_qty))
            return

        # Calculate take-profit price
        sell_price = calculate_take_profit_price(
            TakeProfitParams(
                average_price=avg_price,
                profit_percent=self._settings.profit_percent,
                price_precision=sym_info.price_precision,
            )
        )

        logger.info(
            "updating_sell_order",
            avg_price=str(avg_price),
            net_qty=str(net_qty),
            sell_price=str(sell_price),
            filled_count=len(filled),
        )

        # Cancel existing SELL if present
        if self._cycle.sell_order and self._cycle.sell_order.is_active:
            sell_id = self._cycle.sell_order.binance_order_id
            if sell_id is not None:
                try:
                    await self._client.cancel_order(self._cycle.symbol, sell_id)
                    logger.info("previous_sell_canceled", order_id=sell_id)
                except BinanceClientError as exc:
                    logger.warning("cancel_sell_failed", error=str(exc))

        # Place new SELL order
        new_sell = Order(
            symbol=self._cycle.symbol,
            side=OrderSide.SELL,
            price=sell_price,
            original_qty=net_qty,
        )
        try:
            response = await self._client.place_limit_sell(
                symbol=new_sell.symbol,
                price=new_sell.price,
                quantity=new_sell.original_qty,
                client_order_id=new_sell.client_order_id,
            )
            new_sell = new_sell.model_copy_updated(
                binance_order_id=response["orderId"],
                status=OrderStatus.OPEN,
            )
            logger.info(
                "sell_order_placed",
                price=str(sell_price),
                qty=str(net_qty),
                order_id=response["orderId"],
            )
        except BinanceClientError as exc:
            logger.error("sell_order_failed", error=str(exc))
            new_sell = new_sell.model_copy_updated(status=OrderStatus.REJECTED)

        self._cycle.sell_order = new_sell

    async def _check_grid_shift(self) -> None:
        """
        Detect and handle a grid shift condition.

        If the market has risen past the first BUY level by more than the
        configured threshold, we cancel all pending BUYs, keep any
        already-filled orders, recalculate the grid, and re-place orders.
        """
        assert self._cycle is not None

        # Only shift if no orders have filled yet (pure repositioning)
        if self._cycle.filled_buy_orders:
            return
        if self._cycle.first_buy_price is None:
            return

        market_price = await self._client.get_symbol_price(self._cycle.symbol)

        if not should_shift_grid(
            first_buy_price=self._cycle.first_buy_price,
            current_market_price=market_price,
            threshold_percent=self._settings.grid_shift_threshold_percent,
        ):
            return

        logger.info(
            "grid_shift_detected",
            first_buy_price=str(self._cycle.first_buy_price),
            market_price=str(market_price),
        )

        # Cancel all active BUY orders
        await self._cancel_buy_orders()

        # Re-anchor grid around new market price
        sym_info = await self._client.get_symbol_info(self._cycle.symbol)
        grid_config = GridConfig(
            market_price=market_price,
            initial_offset_percent=self._settings.initial_offset_percent,
            grid_range_percent=self._settings.grid_range_percent,
            grid_levels=self._settings.grid_levels,
            deposit_amount=self._settings.deposit_amount,
            volume_scale_percent=self._settings.volume_scale_percent,
            price_precision=sym_info.price_precision,
            qty_precision=sym_info.qty_precision,
        )

        try:
            new_grid = build_grid(grid_config)
        except ValueError as exc:
            logger.error("grid_rebuild_failed", error=str(exc))
            self._cycle.status = CycleStatus.ABORTED
            return

        self._cycle.grid_levels = new_grid
        self._cycle.first_buy_price = new_grid[0].price

        logger.info(
            "grid_shifted",
            new_top_price=str(new_grid[0].price),
            new_bottom_price=str(new_grid[-1].price),
        )

        await self._place_buy_orders()

    async def _fetch_order_commission(
        self, symbol: str, order_id: Optional[int], base_asset: str = ""
    ) -> Decimal:
        """
        Return total base-asset commission for a filled order.

        Calls GET /api/v3/myTrades with orderId.  Returns Decimal("0") on any
        error so a commission fetch failure never blocks the sell placement.

        Args:
            base_asset: If provided, only sums commissions paid in this asset
                        (e.g. "BTC"). Filters out BNB commissions which must
                        NOT be subtracted from base-asset quantity.
        """
        if order_id is None:
            return Decimal("0")
        try:
            trades = await self._client.get_order_trades(symbol, order_id)
            return sum(
                (
                    Decimal(t.get("commission", "0"))
                    for t in trades
                    if not base_asset or t.get("commissionAsset", "") == base_asset
                ),
                Decimal("0"),
            )
        except BinanceClientError as exc:
            logger.warning(
                "commission_fetch_failed",
                order_id=order_id,
                error=str(exc),
            )
            return Decimal("0")

    async def _cancel_buy_orders(self) -> None:
        """Cancel all active (unfilled) BUY orders in the current cycle."""
        assert self._cycle is not None
        for i, order in enumerate(self._cycle.buy_orders):
            if not order.is_active or order.binance_order_id is None:
                continue
            try:
                await self._client.cancel_order(order.symbol, order.binance_order_id)
                self._cycle.buy_orders[i] = order.model_copy_updated(
                    status=OrderStatus.CANCELED
                )
            except BinanceClientError as exc:
                logger.warning(
                    "cancel_buy_failed",
                    order_id=order.binance_order_id,
                    error=str(exc),
                )

    async def _cancel_all_orders(self) -> None:
        """Cancel every open order (BUYs + SELL) — used on shutdown."""
        try:
            await self._client.cancel_all_open_orders(self._settings.symbol)
        except BinanceClientError as exc:
            logger.error("cancel_all_orders_failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    async def _validate_pre_start(self) -> None:
        """
        Perform pre-flight checks before the bot starts.

        Steps (all modes):
          1. Cancel any open orders left from a previous session (orphan cleanup).
          2. Fetch symbol info and current market price.
          3. Build a dry grid and verify every level meets Binance min notional.

        Steps (live mode only):
          4. Check the account has enough quote-asset balance.

        Raises:
            ValueError:              If grid config is invalid or any order < min notional.
            InsufficientBalanceError: If account balance < deposit_amount.
        """
        s = self._settings

        # 1. Orphan cleanup — cancel any open orders from a previous bot session.
        #    In dry-run mode cancel_all_open_orders is a no-op that returns [].
        _max_cleanup_attempts = 3
        _last_cleanup_exc: Optional[BinanceClientError] = None
        for _attempt in range(1, _max_cleanup_attempts + 1):
            try:
                cancelled = await self._client.cancel_all_open_orders(s.symbol)
                if cancelled:
                    logger.warning(
                        "orphaned_orders_cancelled",
                        count=len(cancelled),
                        symbol=s.symbol,
                        message="Open orders from a previous session were cancelled before starting.",
                    )
                _last_cleanup_exc = None
                break
            except BinanceClientError as exc:
                _last_cleanup_exc = exc
                logger.warning(
                    "orphan_cleanup_attempt_failed",
                    attempt=_attempt,
                    max_attempts=_max_cleanup_attempts,
                    error=str(exc),
                )
                if _attempt < _max_cleanup_attempts:
                    await asyncio.sleep(2.0)
        if _last_cleanup_exc is not None:
            logger.error("orphan_cleanup_all_attempts_failed", error=str(_last_cleanup_exc))
            raise RuntimeError(
                f"Cannot start: failed to cancel existing open orders for {s.symbol} "
                f"after {_max_cleanup_attempts} attempts. "
                "Please cancel them manually on Binance and try again."
            ) from _last_cleanup_exc

        # 2. Fetch symbol info and market price (needed for validation regardless of dry_run).
        sym_info = await self._client.get_symbol_info(s.symbol)
        market_price = await self._client.get_symbol_price(s.symbol)
        self._last_price = market_price

        # 3. Min-notional check — build a trial grid and ensure every level is
        #    above Binance's minimum order value (usually 10 USDT).
        grid_config = GridConfig(
            market_price=market_price,
            initial_offset_percent=s.initial_offset_percent,
            grid_range_percent=s.grid_range_percent,
            grid_levels=s.grid_levels,
            deposit_amount=s.deposit_amount,
            volume_scale_percent=s.volume_scale_percent,
            price_precision=sym_info.price_precision,
            qty_precision=sym_info.qty_precision,
        )
        try:
            trial_grid = build_grid(grid_config)
        except ValueError as exc:
            raise ValueError(f"Invalid grid configuration: {exc}") from exc

        min_order_usdt = min(level.quote_amount for level in trial_grid)
        if min_order_usdt < sym_info.min_notional:
            raise ValueError(
                f"Smallest grid order ({min_order_usdt:.4f} USDT) is below Binance "
                f"minimum notional ({sym_info.min_notional:.2f} USDT). "
                "Reduce 'Orders count' or increase 'Deposit amount'."
            )

        logger.info(
            "grid_validated",
            levels=len(trial_grid),
            min_order_usdt=str(min_order_usdt),
            min_notional=str(sym_info.min_notional),
        )

        # 4. Balance check — skipped in dry-run mode.
        if s.dry_run:
            logger.info("dry_run_skipping_balance_check")
            return

        quote_asset = sym_info.quote_asset
        balance = await self._client.get_account_balance(quote_asset)

        logger.info(
            "balance_check",
            asset=quote_asset,
            balance=str(balance),
            required=str(s.deposit_amount),
        )

        if balance < s.deposit_amount:
            raise InsufficientBalanceError(
                f"Insufficient {quote_asset} balance. "
                f"Have: {balance}, Need: {s.deposit_amount}"
            )

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_order_response(order: Order, raw: dict) -> Order:
        """
        Map a raw Binance order response onto our domain Order model.

        Handles both live and dry-run responses gracefully.
        """
        status_map = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        new_status = status_map.get(raw.get("status", ""), order.status)
        executed_qty = Decimal(raw.get("executedQty", str(order.executed_qty)))

        # Commission: best effort — may not be in polling response
        commission = order.commission
        fills = raw.get("fills", [])
        if fills:
            commission = sum(Decimal(f.get("commission", "0")) for f in fills)

        return order.model_copy_updated(
            status=new_status,
            executed_qty=executed_qty,
            commission=commission,
        )
