"""
Shared pytest configuration and fixtures.
"""

import os

import pytest


# Ensure we always use test settings during tests
os.environ.setdefault("BINANCE_API_KEY", "test_api_key")
os.environ.setdefault("BINANCE_API_SECRET", "test_api_secret")
os.environ.setdefault("ENVIRONMENT", "testnet")
os.environ.setdefault("DRY_RUN", "true")


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the settings singleton cache between tests."""
    from config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
