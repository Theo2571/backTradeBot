"""
Unit tests for the domain calculation module.

These tests exercise every mathematical function in isolation.
No Binance API, no FastAPI — pure Python + Decimal arithmetic.
"""

from decimal import Decimal

import pytest

from domain.calculations import (
    GridConfig,
    TakeProfitParams,
    build_grid,
    calculate_average_price,
    calculate_net_quantity,
    calculate_take_profit_price,
    should_shift_grid,
)


# ======================================================================
# Helpers
# ======================================================================

def make_config(**overrides) -> GridConfig:
    defaults = dict(
        market_price=Decimal("100"),
        initial_offset_percent=Decimal("1"),
        grid_range_percent=Decimal("5"),
        grid_levels=5,
        deposit_amount=Decimal("1000"),
        volume_scale_percent=Decimal("1.5"),
        price_precision=2,
        qty_precision=6,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


# ======================================================================
# Grid price tests
# ======================================================================

class TestGridPrices:
    def test_first_price_is_below_market(self):
        cfg = make_config(market_price=Decimal("100"), initial_offset_percent=Decimal("1"))
        from domain.calculations import calculate_grid_prices
        prices = calculate_grid_prices(cfg)
        assert prices[0] < Decimal("100")
        assert prices[0] <= Decimal("99")  # 1% below 100

    def test_prices_are_descending(self):
        from domain.calculations import calculate_grid_prices
        cfg = make_config()
        prices = calculate_grid_prices(cfg)
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], f"Price[{i}] {prices[i]} not > Price[{i+1}] {prices[i+1]}"

    def test_price_count_matches_grid_levels(self):
        from domain.calculations import calculate_grid_prices
        for n in (2, 5, 10, 20):
            cfg = make_config(grid_levels=n)
            prices = calculate_grid_prices(cfg)
            assert len(prices) == n

    def test_single_level_raises(self):
        from domain.calculations import calculate_grid_prices
        with pytest.raises(ValueError, match="at least 2"):
            calculate_grid_prices(make_config(grid_levels=1))

    def test_last_price_matches_range(self):
        """P_last should be P1 * (1 - range%)."""
        from domain.calculations import calculate_grid_prices
        cfg = make_config(
            market_price=Decimal("100"),
            initial_offset_percent=Decimal("0"),  # P1 = 100
            grid_range_percent=Decimal("10"),
            grid_levels=2,
            price_precision=8,
        )
        prices = calculate_grid_prices(cfg)
        # P1 = 100, P_last = 100 * 0.9 = 90
        assert prices[0] == Decimal("100.00000000")
        assert prices[-1] == Decimal("90.00000000")


# ======================================================================
# Grid volume tests
# ======================================================================

class TestGridVolumes:
    def test_total_cost_does_not_exceed_deposit(self):
        from domain.calculations import calculate_grid_prices, calculate_grid_volumes
        cfg = make_config()
        prices = calculate_grid_prices(cfg)
        volumes = calculate_grid_volumes(cfg)
        total_cost = sum(p * v for p, v in zip(prices, volumes))
        assert total_cost <= cfg.deposit_amount

    def test_usdt_allocation_grows_geometrically(self):
        """
        Per spec point 7: quote-currency spend per level must grow by scale^i.
        We verify the ratio between consecutive USDT allocations is
        approximately equal to the scale multiplier.
        """
        from domain.calculations import calculate_grid_prices, calculate_grid_volumes
        scale = Decimal("1.5")
        cfg = make_config(volume_scale_percent=scale, qty_precision=8)
        prices = calculate_grid_prices(cfg)
        volumes = calculate_grid_volumes(cfg)
        usdt = [prices[i] * volumes[i] for i in range(len(volumes))]
        for i in range(1, len(usdt)):
            ratio = usdt[i] / usdt[i - 1]
            # Allow ±2% tolerance for price-step and rounding effects
            assert abs(ratio - scale) < Decimal("0.02"), (
                f"USDT ratio at level {i}: {ratio} (expected ~{scale})"
            )

    def test_quantities_increase_with_scale(self):
        """With scale > 1, each level buys more base asset than the previous."""
        from domain.calculations import calculate_grid_volumes
        cfg = make_config(volume_scale_percent=Decimal("1.5"))
        volumes = calculate_grid_volumes(cfg)
        for i in range(len(volumes) - 1):
            assert volumes[i] <= volumes[i + 1]

    def test_equal_usdt_when_scale_is_one(self):
        """scale=1.0 means equal USDT per level (not equal base-asset quantity)."""
        from domain.calculations import calculate_grid_prices, calculate_grid_volumes
        cfg = make_config(volume_scale_percent=Decimal("1.0"), qty_precision=8)
        prices = calculate_grid_prices(cfg)
        volumes = calculate_grid_volumes(cfg)
        usdt = [prices[i] * volumes[i] for i in range(len(volumes))]
        # All USDT allocations should be approximately equal
        for i in range(1, len(usdt)):
            assert abs(usdt[i] - usdt[0]) < Decimal("0.01"), (
                f"USDT[{i}]={usdt[i]} differs from USDT[0]={usdt[0]}"
            )

    def test_zero_deposit_raises(self):
        from domain.calculations import calculate_grid_volumes
        with pytest.raises(ValueError):
            calculate_grid_volumes(make_config(deposit_amount=Decimal("0")))


# ======================================================================
# build_grid integration tests
# ======================================================================

class TestBuildGrid:
    def test_returns_correct_number_of_levels(self):
        grid = build_grid(make_config(grid_levels=5))
        assert len(grid) == 5

    def test_grid_index_is_sequential(self):
        grid = build_grid(make_config())
        for i, level in enumerate(grid):
            assert level.index == i

    def test_total_cost_within_deposit(self):
        cfg = make_config(deposit_amount=Decimal("1000"))
        grid = build_grid(cfg)
        total = sum(l.quote_amount for l in grid)
        assert total <= Decimal("1000")

    def test_quote_amount_matches_price_times_qty(self):
        grid = build_grid(make_config())
        for level in grid:
            expected = level.price * level.quantity
            assert abs(level.quote_amount - expected) < Decimal("0.000001")


# ======================================================================
# Average price tests
# ======================================================================

class TestAveragePrice:
    def test_single_order(self):
        avg = calculate_average_price([(Decimal("100"), Decimal("1"))])
        assert avg == Decimal("100")

    def test_two_equal_orders(self):
        avg = calculate_average_price([
            (Decimal("100"), Decimal("1")),
            (Decimal("90"), Decimal("1")),
        ])
        assert avg == Decimal("95")

    def test_weighted_average(self):
        # 100 * 0.5 + 95 * 1.0 = 50 + 95 = 145 / 1.5 = 96.666...
        avg = calculate_average_price([
            (Decimal("100"), Decimal("0.5")),
            (Decimal("95"), Decimal("1.0")),
        ])
        expected = Decimal("145") / Decimal("1.5")
        assert abs(avg - expected) < Decimal("0.0001")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            calculate_average_price([])

    def test_zero_qty_raises(self):
        with pytest.raises(ValueError):
            calculate_average_price([(Decimal("100"), Decimal("0"))])


# ======================================================================
# Net quantity tests
# ======================================================================

class TestNetQuantity:
    def test_basic_commission(self):
        net = calculate_net_quantity([
            (Decimal("1.0"), Decimal("0.001")),
            (Decimal("2.0"), Decimal("0.002")),
        ])
        assert net == Decimal("2.997")

    def test_no_commission(self):
        net = calculate_net_quantity([
            (Decimal("1.0"), Decimal("0")),
            (Decimal("2.0"), Decimal("0")),
        ])
        assert net == Decimal("3.0")


# ======================================================================
# Take profit tests
# ======================================================================

class TestTakeProfit:
    def test_basic_take_profit(self):
        params = TakeProfitParams(
            average_price=Decimal("100"),
            profit_percent=Decimal("2"),
            price_precision=2,
        )
        price = calculate_take_profit_price(params)
        # 100 * 1.02 = 102
        assert price == Decimal("102.00")

    def test_precision_applied(self):
        params = TakeProfitParams(
            average_price=Decimal("95.333333"),
            profit_percent=Decimal("2"),
            price_precision=2,
        )
        price = calculate_take_profit_price(params)
        # Should be truncated (ROUND_DOWN), not rounded
        assert price == Decimal("97.23")  # 97.24 would be round-up

    def test_zero_avg_price_raises(self):
        with pytest.raises(ValueError):
            calculate_take_profit_price(
                TakeProfitParams(Decimal("0"), Decimal("2"), 2)
            )

    def test_zero_profit_raises(self):
        with pytest.raises(ValueError):
            calculate_take_profit_price(
                TakeProfitParams(Decimal("100"), Decimal("0"), 2)
            )


# ======================================================================
# Grid shift detection tests
# ======================================================================

class TestGridShift:
    def test_no_shift_when_price_below_threshold(self):
        assert not should_shift_grid(
            first_buy_price=Decimal("99"),
            current_market_price=Decimal("99.5"),
            threshold_percent=Decimal("1"),
        )

    def test_shift_when_price_above_threshold(self):
        # Market moved 2% above first buy price → should shift
        assert should_shift_grid(
            first_buy_price=Decimal("100"),
            current_market_price=Decimal("102.1"),
            threshold_percent=Decimal("2"),
        )

    def test_no_shift_exactly_at_threshold(self):
        # Exactly at threshold — should NOT shift (strictly greater than)
        assert not should_shift_grid(
            first_buy_price=Decimal("100"),
            current_market_price=Decimal("102"),
            threshold_percent=Decimal("2"),
        )

    def test_no_shift_when_price_below_first_buy(self):
        """Market dropped below our grid — normal scenario, no shift needed."""
        assert not should_shift_grid(
            first_buy_price=Decimal("99"),
            current_market_price=Decimal("97"),
            threshold_percent=Decimal("1"),
        )

    def test_no_shift_on_zero_prices(self):
        assert not should_shift_grid(
            first_buy_price=Decimal("0"),
            current_market_price=Decimal("100"),
            threshold_percent=Decimal("1"),
        )

    def test_no_shift_when_threshold_is_zero(self):
        """threshold=0 means grid shift is disabled regardless of price."""
        assert not should_shift_grid(
            first_buy_price=Decimal("99"),
            current_market_price=Decimal("150"),
            threshold_percent=Decimal("0"),
        )


# ======================================================================
# Decimal purity — no float contamination anywhere
# ======================================================================

class TestDecimalPurity:
    """
    Verify that no float leaks out of the calculation layer.

    Floats are unacceptable in financial code because binary rounding
    errors accumulate across operations and can silently distort P&L.
    Every value that leaves this module must be a Decimal.
    """

    def test_grid_prices_are_decimal(self):
        from domain.calculations import calculate_grid_prices
        prices = calculate_grid_prices(make_config())
        for p in prices:
            assert isinstance(p, Decimal), f"Expected Decimal, got {type(p).__name__}: {p}"

    def test_grid_volumes_are_decimal(self):
        from domain.calculations import calculate_grid_volumes
        vols = calculate_grid_volumes(make_config())
        for v in vols:
            assert isinstance(v, Decimal), f"Expected Decimal, got {type(v).__name__}: {v}"

    def test_build_grid_fields_are_decimal(self):
        grid = build_grid(make_config())
        for level in grid:
            assert isinstance(level.price, Decimal)
            assert isinstance(level.quantity, Decimal)
            assert isinstance(level.quote_amount, Decimal)

    def test_average_price_is_decimal(self):
        avg = calculate_average_price([
            (Decimal("100"), Decimal("1")),
            (Decimal("90"),  Decimal("2")),
        ])
        assert isinstance(avg, Decimal)

    def test_take_profit_is_decimal(self):
        price = calculate_take_profit_price(
            TakeProfitParams(Decimal("100"), Decimal("2"), 2)
        )
        assert isinstance(price, Decimal)

    def test_net_quantity_is_decimal(self):
        from domain.calculations import calculate_net_quantity
        net = calculate_net_quantity([(Decimal("1.0"), Decimal("0.001"))])
        assert isinstance(net, Decimal)


# ======================================================================
# Parametrized grid configuration — coverage across realistic inputs
# ======================================================================

class TestGridParametrized:
    @pytest.mark.parametrize("levels", [2, 3, 5, 10, 20])
    def test_level_count_always_matches_config(self, levels):
        cfg = make_config(grid_levels=levels)
        grid = build_grid(cfg)
        assert len(grid) == levels

    @pytest.mark.parametrize("levels", [2, 3, 5, 10, 20])
    def test_total_cost_within_deposit_for_any_level_count(self, levels):
        cfg = make_config(grid_levels=levels, deposit_amount=Decimal("1000"))
        grid = build_grid(cfg)
        total = sum(l.quote_amount for l in grid)
        assert total <= cfg.deposit_amount, (
            f"Grid cost {total} exceeds deposit {cfg.deposit_amount} for {levels} levels"
        )

    @pytest.mark.parametrize("market_price", [
        Decimal("100"),
        Decimal("1000"),
        Decimal("50000"),
        Decimal("300000"),
    ])
    def test_grid_works_across_price_magnitudes(self, market_price):
        cfg = make_config(market_price=market_price, qty_precision=8)
        grid = build_grid(cfg)
        assert len(grid) == cfg.grid_levels
        assert all(l.price > 0 for l in grid)
        assert all(l.quantity > 0 for l in grid)
        assert all(l.quote_amount > 0 for l in grid)

    @pytest.mark.parametrize("scale", [
        Decimal("1.0"),
        Decimal("1.2"),
        Decimal("1.5"),
        Decimal("2.0"),
    ])
    def test_volume_scale_always_keeps_cost_within_deposit(self, scale):
        cfg = make_config(volume_scale_percent=scale, deposit_amount=Decimal("1000"))
        grid = build_grid(cfg)
        total = sum(l.quote_amount for l in grid)
        assert total <= cfg.deposit_amount


# ======================================================================
# Profit calculation invariants
# ======================================================================

class TestProfitInvariants:
    """
    Cross-function property checks: take-profit price must always produce
    a positive spread over average buy price, regardless of inputs.
    """

    @pytest.mark.parametrize("avg_price,profit_pct", [
        (Decimal("100"),   Decimal("1")),
        (Decimal("50000"), Decimal("2")),
        (Decimal("0.001"), Decimal("5")),
    ])
    def test_sell_price_always_above_avg_buy(self, avg_price, profit_pct):
        params = TakeProfitParams(avg_price, profit_pct, price_precision=8)
        sell = calculate_take_profit_price(params)
        assert sell > avg_price, f"Sell {sell} not above avg buy {avg_price}"

    def test_higher_profit_percent_gives_higher_sell_price(self):
        base = TakeProfitParams(Decimal("100"), Decimal("1"), 2)
        high = TakeProfitParams(Decimal("100"), Decimal("5"), 2)
        assert calculate_take_profit_price(high) > calculate_take_profit_price(base)
