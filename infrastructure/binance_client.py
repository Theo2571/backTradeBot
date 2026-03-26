"""
Binance API wrapper — Infrastructure layer.

Wraps python-binance with:
- Async-compatible execution via asyncio.to_thread
- Automatic retries with exponential backoff
- Structured error handling and logging
- Dry-run mode that short-circuits all mutating calls
- Symbol info caching (filters, precision) to avoid redundant requests
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from functools import lru_cache
from typing import Any, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


class BinanceClientError(Exception):
    """Raised when the Binance API returns an unrecoverable error."""


class InsufficientBalanceError(BinanceClientError):
    """Raised when account balance is too low for the requested operation."""


class SymbolInfo:
    """
    Cached exchange info for a single trading pair.

    Attributes:
        price_precision:   Number of decimal places for price values.
        qty_precision:     Number of decimal places for quantity values.
        min_notional:      Minimum order value in quote currency.
        min_qty:           Minimum order quantity.
        step_size:         Quantity increment.
        tick_size:         Price increment.
    """

    def __init__(self, raw: dict) -> None:
        self._raw = raw
        filters = {f["filterType"]: f for f in raw.get("filters", [])}

        price_filter = filters.get("PRICE_FILTER", {})
        lot_size = filters.get("LOT_SIZE", {})
        notional = filters.get("MIN_NOTIONAL", {})

        self.tick_size: Decimal = Decimal(price_filter.get("tickSize", "0.01"))
        self.step_size: Decimal = Decimal(lot_size.get("stepSize", "0.00001"))
        self.min_qty: Decimal = Decimal(lot_size.get("minQty", "0.00001"))
        self.min_notional: Decimal = Decimal(notional.get("minNotional", "10"))

        # Compute precision from step/tick sizes
        self.price_precision: int = abs(self.tick_size.normalize().as_tuple().exponent)
        self.qty_precision: int = abs(self.step_size.normalize().as_tuple().exponent)

    @property
    def base_asset(self) -> str:
        return self._raw.get("baseAsset", "")

    @property
    def quote_asset(self) -> str:
        return self._raw.get("quoteAsset", "")


class BinanceClient:
    """
    Thin async-friendly wrapper around the synchronous python-binance Client.

    All network I/O is dispatched via asyncio.to_thread so that it does not
    block the FastAPI event loop.

    Args:
        api_key:    Binance API key.
        api_secret: Binance API secret.
        testnet:    When True, connect to testnet.binance.vision.
        dry_run:    When True, no mutating requests are sent to Binance.
        max_retries:       Number of retry attempts on transient errors.
        retry_delay:       Seconds to wait between retries.
    """

    # Error codes that should NOT be retried
    _FATAL_ERROR_CODES = {
        -2010,  # New order rejected
        -1121,  # Invalid symbol
        -1100,  # Illegal characters in parameter
        -1013,  # Filter failure
        -2011,  # Unknown order sent
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        dry_run: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._symbol_cache: dict[str, SymbolInfo] = {}

        tld = "com"
        self._client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            tld=tld,
        )
        logger.info(
            "binance_client_initialized",
            testnet=testnet,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _call(self, fn, *args, **kwargs) -> Any:
        """
        Execute a synchronous python-binance call in a thread,
        with retry logic for transient errors.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = await asyncio.to_thread(fn, *args, **kwargs)
                return result

            except BinanceAPIException as exc:
                if exc.code in self._FATAL_ERROR_CODES:
                    logger.error(
                        "binance_fatal_error",
                        code=exc.code,
                        message=exc.message,
                    )
                    raise BinanceClientError(f"Binance error {exc.code}: {exc.message}") from exc

                logger.warning(
                    "binance_transient_error",
                    attempt=attempt,
                    code=exc.code,
                    message=exc.message,
                )
                last_exc = exc

            except BinanceOrderException as exc:
                logger.error("binance_order_exception", code=exc.code, message=exc.message)
                raise BinanceClientError(f"Order error {exc.code}: {exc.message}") from exc

            except Exception as exc:
                logger.warning("binance_unexpected_error", attempt=attempt, error=str(exc))
                last_exc = exc

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)

        raise BinanceClientError(f"Binance call failed after {self.max_retries} attempts") from last_exc

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #

    async def get_symbol_price(self, symbol: str) -> Decimal:
        """Fetch the latest spot price for a symbol."""
        data = await self._call(self._client.get_symbol_ticker, symbol=symbol)
        return Decimal(data["price"])

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Return cached SymbolInfo for the given pair.

        SymbolInfo is fetched once and cached for the lifetime of the client.
        """
        if symbol not in self._symbol_cache:
            raw = await self._call(self._client.get_symbol_info, symbol)
            if raw is None:
                raise BinanceClientError(f"Symbol {symbol!r} not found on Binance")
            self._symbol_cache[symbol] = SymbolInfo(raw)
            logger.debug(
                "symbol_info_cached",
                symbol=symbol,
                price_precision=self._symbol_cache[symbol].price_precision,
                qty_precision=self._symbol_cache[symbol].qty_precision,
            )
        return self._symbol_cache[symbol]

    async def get_account_balance(self, asset: str) -> Decimal:
        """
        Return the free balance for the given asset.

        Args:
            asset: e.g. "USDT" or "BTC"

        Returns:
            Free (available) balance as Decimal.
        """
        account = await self._call(self._client.get_account)
        balances = {b["asset"]: Decimal(b["free"]) for b in account["balances"]}
        return balances.get(asset.upper(), Decimal("0"))

    # ------------------------------------------------------------------ #
    # Order management
    # ------------------------------------------------------------------ #

    async def place_limit_buy(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        client_order_id: str,
    ) -> dict:
        """
        Place a LIMIT BUY order on Binance.

        Args:
            symbol:          Trading pair.
            price:           Limit price.
            quantity:        Base-asset quantity.
            client_order_id: Unique client-side order ID for idempotency.

        Returns:
            Raw Binance order response dict.

        In dry-run mode, returns a simulated response without network call.
        """
        if self.dry_run:
            return self._simulate_order(symbol, "BUY", price, quantity, client_order_id)

        logger.info(
            "placing_limit_buy",
            symbol=symbol,
            price=str(price),
            quantity=str(quantity),
            client_order_id=client_order_id,
        )
        return await self._call(
            self._client.order_limit_buy,
            symbol=symbol,
            quantity=str(quantity),
            price=str(price),
            newClientOrderId=client_order_id,
            timeInForce="GTC",
        )

    async def place_limit_sell(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        client_order_id: str,
    ) -> dict:
        """
        Place a LIMIT SELL order on Binance.

        In dry-run mode, returns a simulated response without network call.
        """
        if self.dry_run:
            return self._simulate_order(symbol, "SELL", price, quantity, client_order_id)

        logger.info(
            "placing_limit_sell",
            symbol=symbol,
            price=str(price),
            quantity=str(quantity),
            client_order_id=client_order_id,
        )
        return await self._call(
            self._client.order_limit_sell,
            symbol=symbol,
            quantity=str(quantity),
            price=str(price),
            newClientOrderId=client_order_id,
            timeInForce="GTC",
        )

    async def cancel_order(self, symbol: str, order_id: int) -> dict:
        """
        Cancel an open order by its Binance order ID.

        Returns the cancellation response. Does not raise if order is
        already filled or canceled (handles -2011 gracefully).
        """
        if self.dry_run:
            logger.debug("dry_run_cancel_order", symbol=symbol, order_id=order_id)
            return {"orderId": order_id, "status": "CANCELED"}

        try:
            return await self._call(
                self._client.cancel_order,
                symbol=symbol,
                orderId=order_id,
            )
        except BinanceClientError as exc:
            if "-2011" in str(exc) or "Unknown order" in str(exc):
                logger.warning(
                    "cancel_order_already_gone",
                    symbol=symbol,
                    order_id=order_id,
                )
                return {"orderId": order_id, "status": "CANCELED"}
            raise

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        """Fetch the current status of an order from Binance."""
        if self.dry_run:
            return {"orderId": order_id, "status": "NEW", "executedQty": "0"}

        return await self._call(
            self._client.get_order,
            symbol=symbol,
            orderId=order_id,
        )

    async def get_order_trades(self, symbol: str, order_id: int) -> list[dict]:
        """
        Fetch all trade fills for a specific order via GET /api/v3/myTrades.

        This is the only reliable way to get commission data for limit orders
        that filled passively (resting on the book) — the GET /api/v3/order
        endpoint does NOT return fills, only the initial POST does.

        Args:
            symbol:   Trading pair (e.g. "BTCUSDT").
            order_id: Binance order ID.

        Returns:
            List of trade dicts, each containing 'commission', 'commissionAsset', etc.
        """
        if self.dry_run:
            return []

        return await self._call(
            self._client.get_my_trades,
            symbol=symbol,
            orderId=order_id,
        )

    async def cancel_all_open_orders(self, symbol: str) -> list[dict]:
        """
        Cancel every open order for the given symbol.

        Used during graceful shutdown and grid shifts.
        Returns list of cancellation responses.
        """
        if self.dry_run:
            logger.info("dry_run_cancel_all_orders", symbol=symbol)
            return []

        open_orders = await self._call(
            self._client.get_open_orders,
            symbol=symbol,
        )
        if not open_orders:
            return []

        logger.info("canceling_all_open_orders", symbol=symbol, count=len(open_orders))
        results = []
        for order in open_orders:
            try:
                result = await self.cancel_order(symbol, order["orderId"])
                results.append(result)
            except BinanceClientError as exc:
                logger.warning(
                    "failed_to_cancel_order",
                    order_id=order["orderId"],
                    error=str(exc),
                )
        return results

    # ------------------------------------------------------------------ #
    # Dry-run simulation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _simulate_order(
        symbol: str,
        side: str,
        price: Decimal,
        quantity: Decimal,
        client_order_id: str,
    ) -> dict:
        """Build a fake Binance order response for dry-run mode."""
        fake_id = int(time.time() * 1000) % 1_000_000_000
        logger.info(
            "dry_run_order_simulated",
            symbol=symbol,
            side=side,
            price=str(price),
            quantity=str(quantity),
            fake_id=fake_id,
        )
        return {
            "symbol": symbol,
            "orderId": fake_id,
            "clientOrderId": client_order_id,
            "price": str(price),
            "origQty": str(quantity),
            "executedQty": "0.00000000",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": side,
            "transactTime": int(time.time() * 1000),
        }


def create_binance_client() -> BinanceClient:
    """
    Factory that creates a BinanceClient from application settings.

    This is the canonical way to instantiate the client — it ensures
    settings are loaded once and the correct environment is used.
    """
    settings = get_settings()
    return BinanceClient(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        testnet=settings.is_testnet,
        dry_run=settings.dry_run,
        max_retries=settings.api_max_retries,
        retry_delay=settings.api_retry_delay_seconds,
    )
