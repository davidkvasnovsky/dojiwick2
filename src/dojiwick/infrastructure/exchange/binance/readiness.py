"""Binance adapter readiness guard — validates API credentials from environment."""

import os

from dojiwick.domain.errors import ConfigurationError


def assert_binance_ready(*, api_key_env: str, api_secret_env: str) -> tuple[str, str]:
    """Validate Binance API credentials are present in the environment.

    Returns the resolved (api_key, api_secret) tuple on success.
    Raises ConfigurationError if either env var is empty or missing.
    """
    api_key = os.environ.get(api_key_env, "").strip()
    api_secret = os.environ.get(api_secret_env, "").strip()
    if not api_key:
        raise ConfigurationError(f"Binance API key not set: env var {api_key_env} is empty or missing")
    if not api_secret:
        raise ConfigurationError(f"Binance API secret not set: env var {api_secret_env} is empty or missing")
    return api_key, api_secret
