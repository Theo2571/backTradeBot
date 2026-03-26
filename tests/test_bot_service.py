"""
Unit tests for BotService business logic.

BinanceClient is fully mocked — no network calls.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from config.settings import Environment, Settings
from domain.models import BotStatus, CycleStatus, DcaCycle, GridLevel, Order, OrderSide, OrderStatus
from infrastructure.binance_client import BinanceClientError, InsufficientBalanceError
from services.bot_service import BotService


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def settings() -> Settings:
    return Settings(
        binance_api_key="test_key",
        binance_api_secret="test_secret",
        environment=Environment.TESTNET,
        dry_run=True,
        symbol="BTCUSDT",
        deposit_amount=Decimal("1000"),
        grid_levels=3,
        initial_offset_percent=Decimal("1"),
        grid_range_percent=Decimal("5"),
        volume_scale_percent=Decimal("1.5"),
        profit_percent=Decimal("2"),
        grid_shift_threshold_percent=Decimal("1"),
        poll_interval_seconds=0.01,  # Fast for tests
        api_max_retries=1,
        api_retry_delay_seconds=0,
    )


@pytest.fixture
def mock_binance_client() -> MagicMock:
    client = MagicMock()
    client.dry_run = True

    # Symbol info
    sym_info = MagicMock()
    sym_info.price_precision = 2
    sym_info.qty_precision = 6
    sym_info.quote_asset = "USDT"
    sym_info.base_asset = "BTC"
    sym_info.min_notional = Decimal("10")
    client.get_symbol_info = AsyncMock(return_value=sym_info)

    # Market price
    client.get_symbol_price = AsyncMock(return_value=Decimal("50000"))

    # Balance
    client.get_account_balance = AsyncMock(return_value=Decimal("10000"))

    # Order placement
    def fake_buy(symbol, price, quantity, client_order_id):
        import time
        return {
            "orderId": int(time.time() * 1000) % 1_000_000,
            "clientOrderId": client_order_id,
            "status": "NEW",
            "executedQty": "0",
        }

    client.place_limit_buy = AsyncMock(side_effect=fake_buy)
    client.place_limit_sell = AsyncMock(side_effect=lambda **kwargs: {
        "orderId": 999999,
        "clientOrderId": "sell_test",
        "status": "NEW",
        "executedQty": "0",
    })
    client.cancel_order = AsyncMock(return_value={"status": "CANCELED"})
    client.cancel_all_open_orders = AsyncMock(return_value=[])
    client.get_order_trades = AsyncMock(return_value=[])

    # Default get_order_status → always NEW (unfilled)
    client.get_order_status = AsyncMock(return_value={
        "status": "NEW",
        "executedQty": "0",
        "fills": [],
    })

    return client


@pytest.fixture
def bot_service(settings: Settings, mock_binance_client: MagicMock) -> BotService:
    return BotService(settings=settings, client=mock_binance_client)


# ======================================================================
# Status tests
# ======================================================================

class TestBotStatus:
    def test_initial_status_is_idle(self, bot_service: BotService):
        assert bot_service._status == BotStatus.IDLE

    def test_get_status_returns_dict(self, bot_service: BotService):
        status = bot_service.get_status()
        assert status["status"] == "stopped"
        assert status["cyclesCompleted"] == 0
        assert status["activeOrders"] == []
        assert status["executedOrders"] == []
        assert status["currentPrice"] == 0.0

    def test_apply_credentials_while_running_preserves_cycle_state(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        """POST /verify-credentials must not wipe the bot when it is already running."""
        bot_service._status = BotStatus.RUNNING
        bot_service._cycles_completed = 3
        old_client = bot_service._client
        bot_service.apply_exchange_credentials("new_key", "new_secret", True)
        assert bot_service._status == BotStatus.RUNNING
        assert bot_service._cycles_completed == 3
        assert bot_service._client is not old_client


# ======================================================================
# Validation tests
# ======================================================================

class TestPreStartValidation:
    @pytest.mark.asyncio
    async def test_dry_run_skips_balance_check(
        self, settings: Settings, mock_binance_client: MagicMock
    ):
        settings.__dict__["dry_run"] = True
        bot = BotService(settings=settings, client=mock_binance_client)
        await bot._validate_pre_start()
        mock_binance_client.get_account_balance.assert_not_called()

    @pytest.mark.asyncio
    async def test_insufficient_balance_raises(
        self, settings: Settings, mock_binance_client: MagicMock
    ):
        settings.__dict__["dry_run"] = False
        mock_binance_client.get_account_balance = AsyncMock(return_value=Decimal("500"))
        bot = BotService(settings=settings, client=mock_binance_client)

        with pytest.raises(InsufficientBalanceError, match="Insufficient"):
            await bot._validate_pre_start()


# ======================================================================
# Start / Stop tests
# ======================================================================

class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_changes_status_to_running(self, bot_service: BotService):
        await bot_service.start()
        assert bot_service._status == BotStatus.RUNNING
        # Cleanup
        await bot_service.stop()

    @pytest.mark.asyncio
    async def test_double_start_raises(self, bot_service: BotService):
        await bot_service.start()
        with pytest.raises(RuntimeError, match="already running"):
            await bot_service.start()
        await bot_service.stop()

    @pytest.mark.asyncio
    async def test_stop_on_idle_is_noop(self, bot_service: BotService):
        # Should not raise
        await bot_service.stop()
        assert bot_service._status == BotStatus.IDLE

    @pytest.mark.asyncio
    async def test_stop_cancels_orders(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        await bot_service.start()
        await bot_service.stop()
        mock_binance_client.cancel_all_open_orders.assert_called()


# ======================================================================
# Order parsing tests
# ======================================================================

class TestOrderParsing:
    def test_parse_filled_order(self, bot_service: BotService):
        from domain.models import Order, OrderSide, OrderStatus
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
        )
        raw = {
            "status": "FILLED",
            "executedQty": "0.01000000",
            "fills": [{"commission": "0.00001000", "commissionAsset": "BTC"}],
        }
        updated = BotService._parse_order_response(order, raw)
        assert updated.status == OrderStatus.FILLED
        assert updated.executed_qty == Decimal("0.01")
        assert updated.commission == Decimal("0.00001")

    def test_parse_new_order(self, bot_service: BotService):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
        )
        raw = {"status": "NEW", "executedQty": "0"}
        updated = BotService._parse_order_response(order, raw)
        assert updated.status == OrderStatus.OPEN

    def test_parse_partially_filled(self, bot_service: BotService):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
        )
        raw = {"status": "PARTIALLY_FILLED", "executedQty": "0.005"}
        updated = BotService._parse_order_response(order, raw)
        assert updated.status == OrderStatus.PARTIALLY_FILLED
        assert updated.executed_qty == Decimal("0.005")

    def test_parse_expired(self, bot_service: BotService):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
        )
        updated = BotService._parse_order_response(order, {"status": "EXPIRED", "executedQty": "0"})
        assert updated.status == OrderStatus.EXPIRED

    def test_parse_canceled(self, bot_service: BotService):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
        )
        updated = BotService._parse_order_response(order, {"status": "CANCELED", "executedQty": "0"})
        assert updated.status == OrderStatus.CANCELED

    def test_unknown_status_preserves_original(self, bot_service: BotService):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
            status=OrderStatus.OPEN,
        )
        updated = BotService._parse_order_response(order, {"status": "GHOST_STATUS", "executedQty": "0"})
        assert updated.status == OrderStatus.OPEN

    def test_commission_summed_from_multiple_fills(self, bot_service: BotService):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
        )
        raw = {
            "status": "FILLED",
            "executedQty": "0.01",
            "fills": [{"commission": "0.000005"}, {"commission": "0.000005"}],
        }
        updated = BotService._parse_order_response(order, raw)
        assert updated.commission == Decimal("0.00001")


# ======================================================================
# Profit accrual tests
# ======================================================================

def _filled_buy(price: str, qty: str, commission: str = "0") -> Order:
    """Helper: create a fully-filled BUY order."""
    return Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        price=Decimal(price),
        original_qty=Decimal(qty),
        executed_qty=Decimal(qty),
        commission=Decimal(commission),
        status=OrderStatus.FILLED,
        binance_order_id=12345,
    )


def _filled_sell(price: str, qty: str, commission: str = "0") -> Order:
    """Helper: create a fully-filled SELL order."""
    return Order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        price=Decimal(price),
        original_qty=Decimal(qty),
        executed_qty=Decimal(qty),
        commission=Decimal(commission),
        status=OrderStatus.FILLED,
        binance_order_id=99999,
    )


class TestAccrueProfit:
    def test_no_cycle_is_noop(self, bot_service: BotService):
        bot_service._cycle = None
        bot_service._accrue_profit()
        assert bot_service._total_profit == Decimal("0")

    def test_no_sell_order_is_noop(self, bot_service: BotService):
        buy = _filled_buy("50000", "0.01")
        bot_service._cycle = DcaCycle(symbol="BTCUSDT", buy_orders=[buy], sell_order=None)
        bot_service._accrue_profit()
        assert bot_service._total_profit == Decimal("0")

    def test_basic_profit_math(self, bot_service: BotService):
        """revenue = sell_price × net_qty; profit = revenue − total_spent − sell_commission."""
        # buy: spend 500 USDT, receive 0.01 BTC minus 0.00001 BTC commission
        buy = _filled_buy("50000", "0.01", "0.00001")
        # sell: 0.00999 BTC at 51000 USDT, pay 0.5 USDT commission
        sell = _filled_sell("51000", "0.00999", "0.5")

        bot_service._cycle = DcaCycle(symbol="BTCUSDT", buy_orders=[buy], sell_order=sell)
        bot_service._accrue_profit()

        # net_qty   = 0.01 - 0.00001 = 0.00999
        # total_spent = 50000 * 0.01 = 500
        # revenue   = 51000 * 0.00999 = 509.449...
        # profit    = 509.449... - 500 - 0.5 = 8.949...
        expected = Decimal("51000") * Decimal("0.00999") - Decimal("500") - Decimal("0.5")
        assert bot_service._total_profit == expected
        assert bot_service._last_cycle_profit == expected

    def test_profit_accumulates_across_cycles(self, bot_service: BotService):
        """_total_profit is a running sum — each call adds to it."""
        for sell_price, expected_profit in [("51000", Decimal("10")), ("52000", Decimal("20"))]:
            buy = _filled_buy("50000", "0.01")          # cost = 500
            sell = _filled_sell(sell_price, "0.01")     # revenue = price * 0.01
            bot_service._cycle = DcaCycle(symbol="BTCUSDT", buy_orders=[buy], sell_order=sell)
            bot_service._accrue_profit()

        # Cycle 1: 51000*0.01 - 500 = 10; Cycle 2: 52000*0.01 - 500 = 20 → total = 30
        assert bot_service._total_profit == Decimal("30")

    def test_multi_fill_average_price_used(self, bot_service: BotService):
        """When multiple BUY levels fill, profit is calculated from VWAP."""
        buy_1 = _filled_buy("50000", "0.01")   # spend 500
        buy_2 = _filled_buy("48000", "0.01")   # spend 480
        sell = _filled_sell("52000", "0.02")   # revenue = 52000 * 0.02 = 1040

        bot_service._cycle = DcaCycle(
            symbol="BTCUSDT", buy_orders=[buy_1, buy_2], sell_order=sell
        )
        bot_service._accrue_profit()

        # total_spent = 500 + 480 = 980; net_qty = 0.02; revenue = 1040; profit = 60
        assert bot_service._total_profit == Decimal("60")


# ======================================================================
# Buy fill → sell order placement
# ======================================================================

class TestBuyFillTriggersSell:
    @pytest.mark.asyncio
    async def test_filled_buy_triggers_sell_placement(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        active_buy = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.001"),
            status=OrderStatus.OPEN,
            binance_order_id=111,
        )
        bot_service._cycle = DcaCycle(symbol="BTCUSDT", buy_orders=[active_buy])

        mock_binance_client.get_order_status = AsyncMock(return_value={
            "status": "FILLED", "executedQty": "0.001000", "fills": [],
        })

        await bot_service._refresh_buy_orders()

        mock_binance_client.place_limit_sell.assert_called_once()

    @pytest.mark.asyncio
    async def test_unfilled_buy_does_not_trigger_sell(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        active_buy = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.001"),
            status=OrderStatus.OPEN,
            binance_order_id=111,
        )
        bot_service._cycle = DcaCycle(symbol="BTCUSDT", buy_orders=[active_buy])

        mock_binance_client.get_order_status = AsyncMock(return_value={
            "status": "NEW", "executedQty": "0",
        })

        await bot_service._refresh_buy_orders()

        mock_binance_client.place_limit_sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_existing_sell_cancelled_before_replace(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        """A second buy fill must cancel the stale SELL before placing the new one."""
        existing_sell = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            price=Decimal("51000"),
            original_qty=Decimal("0.001"),
            status=OrderStatus.OPEN,
            binance_order_id=999,
        )
        active_buy = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("49000"),
            original_qty=Decimal("0.001"),
            status=OrderStatus.OPEN,
            binance_order_id=222,
        )
        bot_service._cycle = DcaCycle(
            symbol="BTCUSDT", buy_orders=[active_buy], sell_order=existing_sell
        )

        mock_binance_client.get_order_status = AsyncMock(return_value={
            "status": "FILLED", "executedQty": "0.001000", "fills": [],
        })

        await bot_service._refresh_buy_orders()

        mock_binance_client.cancel_order.assert_called_once_with("BTCUSDT", 999)
        mock_binance_client.place_limit_sell.assert_called_once()

    @pytest.mark.asyncio
    async def test_sell_price_reflects_profit_target(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        """Sell price = avg_buy * (1 + profit_percent / 100), rounded down."""
        active_buy = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            original_qty=Decimal("0.01"),
            status=OrderStatus.OPEN,
            binance_order_id=111,
        )
        bot_service._cycle = DcaCycle(symbol="BTCUSDT", buy_orders=[active_buy])

        mock_binance_client.get_order_status = AsyncMock(return_value={
            "status": "FILLED", "executedQty": "0.01", "fills": [],
        })

        await bot_service._refresh_buy_orders()

        # avg_buy=50000, profit=2% → sell_price = 50000 * 1.02 = 51000.00
        call_kwargs = mock_binance_client.place_limit_sell.call_args.kwargs
        assert call_kwargs["price"] == Decimal("51000.00")


# ======================================================================
# Sell fill → cycle completion
# ======================================================================

class TestSellFillCompletesCycle:
    @pytest.mark.asyncio
    async def test_sell_fill_marks_cycle_complete(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        sell = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            price=Decimal("51000"),
            original_qty=Decimal("0.001"),
            status=OrderStatus.OPEN,
            binance_order_id=999,
        )
        bot_service._cycle = DcaCycle(symbol="BTCUSDT", sell_order=sell)

        mock_binance_client.get_order_status = AsyncMock(return_value={
            "status": "FILLED", "executedQty": "0.001000",
        })

        await bot_service._check_sell_order()

        assert bot_service._cycle.status == CycleStatus.COMPLETED
        assert bot_service._cycle.completed_at is not None

    @pytest.mark.asyncio
    async def test_unfilled_sell_keeps_cycle_active(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        sell = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            price=Decimal("51000"),
            original_qty=Decimal("0.001"),
            status=OrderStatus.OPEN,
            binance_order_id=999,
        )
        bot_service._cycle = DcaCycle(symbol="BTCUSDT", sell_order=sell)

        mock_binance_client.get_order_status = AsyncMock(return_value={
            "status": "NEW", "executedQty": "0",
        })

        await bot_service._check_sell_order()

        assert bot_service._cycle.status == CycleStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_no_sell_order_does_not_call_binance(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        bot_service._cycle = DcaCycle(symbol="BTCUSDT", sell_order=None)
        await bot_service._check_sell_order()
        mock_binance_client.get_order_status.assert_not_called()


# ======================================================================
# Grid shift detection and re-placement
# ======================================================================

def _cycle_with_active_buy(first_buy_price: str = "48000") -> DcaCycle:
    """Cycle with one open buy order and no fills — shift can trigger."""
    active_buy = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        price=Decimal(first_buy_price),
        original_qty=Decimal("0.001"),
        status=OrderStatus.OPEN,
        binance_order_id=111,
    )
    dummy_level = GridLevel(
        index=0,
        price=Decimal(first_buy_price),
        quantity=Decimal("0.001"),
        quote_amount=Decimal("48"),
    )
    return DcaCycle(
        symbol="BTCUSDT",
        grid_levels=[dummy_level],
        buy_orders=[active_buy],
        first_buy_price=Decimal(first_buy_price),
    )


class TestGridShiftService:
    @pytest.mark.asyncio
    async def test_shift_cancels_old_orders_and_places_new_grid(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        """
        Market rose >1% above first buy price → cancel existing orders,
        recalculate grid around new market price, place fresh BUY orders.
        """
        bot_service._cycle = _cycle_with_active_buy("48000")
        # 50000 is ~4.2% above 48000, well above the 1% threshold in settings
        mock_binance_client.get_symbol_price = AsyncMock(return_value=Decimal("50000"))

        await bot_service._check_grid_shift()

        # The one open order must be cancelled
        mock_binance_client.cancel_order.assert_called_once_with("BTCUSDT", 111)
        # New grid (3 levels, as per settings fixture) must be placed
        assert mock_binance_client.place_limit_buy.call_count == 3

    @pytest.mark.asyncio
    async def test_no_shift_when_price_below_threshold(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        bot_service._cycle = _cycle_with_active_buy("48000")
        # 48100: deviation = 0.2% < 1% threshold → no shift
        mock_binance_client.get_symbol_price = AsyncMock(return_value=Decimal("48100"))

        await bot_service._check_grid_shift()

        mock_binance_client.cancel_order.assert_not_called()
        mock_binance_client.place_limit_buy.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_shift_once_any_order_has_filled(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        """Grid shift is only valid before any buy has filled; once filled, hold the grid."""
        cycle = _cycle_with_active_buy("48000")
        filled_buy = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("48000"),
            original_qty=Decimal("0.001"),
            executed_qty=Decimal("0.001"),
            status=OrderStatus.FILLED,
            binance_order_id=222,
        )
        cycle.buy_orders.append(filled_buy)
        bot_service._cycle = cycle

        mock_binance_client.get_symbol_price = AsyncMock(return_value=Decimal("50000"))

        await bot_service._check_grid_shift()

        mock_binance_client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_grid_shift_updates_first_buy_price(
        self, bot_service: BotService, mock_binance_client: MagicMock
    ):
        """After a shift, the cycle's reference price must update to the new grid top."""
        bot_service._cycle = _cycle_with_active_buy("48000")
        mock_binance_client.get_symbol_price = AsyncMock(return_value=Decimal("50000"))

        await bot_service._check_grid_shift()

        # New first_buy_price must be below 50000 (offset 1% → ~49500)
        assert bot_service._cycle.first_buy_price < Decimal("50000")
        assert bot_service._cycle.first_buy_price > Decimal("48000")
