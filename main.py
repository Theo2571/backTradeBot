"""
Application entry point.

Wires together the FastAPI app, lifespan events, dependency injection,
and all routers.  Keep this file thin — it should only contain startup/
shutdown orchestration.
"""

from __future__ import annotations

import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.router import router
from config.logging_config import configure_logging, get_logger
from config.settings import get_settings
from infrastructure.binance_client import create_binance_client
from services.bot_service import BotService

# Logger is created after configure_logging() is called in lifespan
logger = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup:
    - Configure logging
    - Load and validate settings
    - Create Binance client
    - Create BotService and attach to app.state

    Shutdown:
    - Stop the bot gracefully (cancel all orders)
    """
    global logger  # noqa: PLW0603

    # ------------------------------------------------------------------ #
    # STARTUP
    # ------------------------------------------------------------------ #
    configure_logging()
    logger = get_logger(__name__)

    settings = get_settings()

    logger.info(
        "application_starting",
        symbol=settings.symbol,
        environment=settings.environment.value,
        dry_run=settings.dry_run,
        log_level=settings.log_level.value,
    )

    if settings.is_mainnet and settings.dry_run:
        logger.warning(
            "mainnet_dry_run_notice",
            message=(
                "Running against MAINNET with DRY_RUN=true. "
                "No real orders will be placed, but real API credentials are used."
            ),
        )

    if settings.is_mainnet and not settings.dry_run:
        logger.warning(
            "MAINNET_LIVE_TRADING_ACTIVE",
            message=(
                "!!! LIVE TRADING ON MAINNET — REAL MONEY WILL BE USED !!!"
            ),
        )

    binance_client = create_binance_client()
    bot_service = BotService(settings=settings, client=binance_client)

    app.state.bot_service = bot_service
    app.state.settings = settings

    logger.info("application_ready")
    yield

    # ------------------------------------------------------------------ #
    # SHUTDOWN
    # ------------------------------------------------------------------ #
    logger.info("application_shutting_down")
    await bot_service.stop()
    logger.info("application_shutdown_complete")


def create_app() -> FastAPI:
    """
    Application factory.

    Returns a configured FastAPI instance.  Calling this as a factory
    (rather than using a module-level app instance) makes it easy to
    create fresh instances in tests.
    """
    settings = get_settings()

    app = FastAPI(
        title="Binance DCA Trading Bot",
        description=(
            "Production-grade Dollar Cost Averaging (DCA) bot for Binance Spot. "
            "Manages a grid of limit BUY orders with automatic take-profit SELL."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # CORS
    # Vite dev server runs on :5173 by default.
    # In production build the frontend is served statically — add your domain.
    # ------------------------------------------------------------------ #
    cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",   # vite preview
        "http://127.0.0.1:4173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # ------------------------------------------------------------------ #
    # Global exception handler
    # ------------------------------------------------------------------ #
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception) -> JSONResponse:
        if logger:
            logger.exception("unhandled_exception", path=str(request.url), error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ------------------------------------------------------------------ #
    # Health check (unauthenticated)
    # ------------------------------------------------------------------ #
    @app.get("/health", include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok"}

    # ------------------------------------------------------------------ #
    # Routers — prefix /api matches the Vite proxy config in frontTradeBot
    # ------------------------------------------------------------------ #
    app.include_router(router, prefix="/api")

    return app


app = create_app()


# ======================================================================
# Development entry point
# ======================================================================

def _handle_signal(signum, frame) -> None:
    """Handle OS signals for clean shutdown in development."""
    print("\nReceived shutdown signal. Stopping...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Never use reload=True in production
        log_level="info",
        access_log=True,
    )
