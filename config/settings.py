"""
Application settings loaded from environment variables.

Uses pydantic-settings for validation and type coercion.
All monetary values are stored as Decimal for precision.
"""

from decimal import Decimal
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Central configuration object for the DCA bot.

    Loaded from environment variables or a .env file.
    All fields are validated at startup — the bot will refuse
    to start if any required value is missing or out of range.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Binance credentials
    # Credentials are supplied per-request via the frontend /start payload.
    # These defaults allow the server to start without real keys in .env.
    # The actual keys are injected into a fresh BinanceClient in the router.
    # ------------------------------------------------------------------ #
    binance_api_key: str = Field(default="placeholder", description="Binance API key")
    binance_api_secret: str = Field(default="placeholder", description="Binance API secret")

    # ------------------------------------------------------------------ #
    # Runtime environment
    # ------------------------------------------------------------------ #
    environment: Environment = Field(
        default=Environment.TESTNET,
        description="testnet or mainnet",
    )
    dry_run: bool = Field(
        default=True,
        description="When True, orders are simulated and never sent to Binance",
    )

    # ------------------------------------------------------------------ #
    # Trading parameters
    # ------------------------------------------------------------------ #
    symbol: str = Field(default="BTCUSDT", description="Binance spot trading pair")
    deposit_amount: Decimal = Field(
        default=Decimal("1000.0"),
        gt=0,
        description="Total quote-currency capital per DCA cycle",
    )
    grid_levels: int = Field(
        default=5,
        ge=2,
        le=100,
        description="Number of grid BUY orders",
    )
    initial_offset_percent: Decimal = Field(
        default=Decimal("1.0"),
        gt=0,
        lt=100,
        description="Distance of first BUY below market price (%)",
    )
    grid_range_percent: Decimal = Field(
        default=Decimal("5.0"),
        gt=0,
        lt=100,
        description="Total price span of the grid below first BUY (%)",
    )
    volume_scale_percent: Decimal = Field(
        default=Decimal("1.5"),
        gt=0,
        description="Multiplier applied to each subsequent grid level volume",
    )
    profit_percent: Decimal = Field(
        default=Decimal("2.0"),
        gt=0,
        lt=100,
        description="Take-profit target above average fill price (%)",
    )

    # ------------------------------------------------------------------ #
    # Safety limits
    # ------------------------------------------------------------------ #
    grid_shift_threshold_percent: Decimal = Field(
        default=Decimal("1.0"),
        gt=0,
        description="Market price rise that triggers grid recalculation (%)",
    )

    # ------------------------------------------------------------------ #
    # Infrastructure tuning
    # ------------------------------------------------------------------ #
    poll_interval_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Seconds between order-status polling iterations",
    )
    api_max_retries: int = Field(
        default=3,
        ge=1,
        description="Max retry attempts for Binance API calls",
    )
    api_retry_delay_seconds: float = Field(
        default=2.0,
        ge=0,
        description="Delay between retry attempts (seconds)",
    )

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging verbosity")
    log_file: Optional[str] = Field(
        default=None,
        description="Optional file path for log output",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("deposit_amount")
    @classmethod
    def deposit_must_be_positive(cls, v: Decimal) -> Decimal:
        """deposit_amount must be > 0; upper bound is the exchange quote balance (checked at start)."""
        if v <= 0:
            raise ValueError("deposit_amount must be positive")
        return v

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_uppercase(cls, v: str) -> str:
        return v.upper().strip()

    @property
    def is_testnet(self) -> bool:
        return self.environment == Environment.TESTNET

    @property
    def is_mainnet(self) -> bool:
        return self.environment == Environment.MAINNET

    @property
    def binance_base_url(self) -> str:
        if self.is_testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached so that environment variables are parsed only once.
    In tests, call get_settings.cache_clear() to reset.
    """
    return Settings()
