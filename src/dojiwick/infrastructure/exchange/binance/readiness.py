"""Binance adapter readiness guard — validates API credentials from environment."""

import os

from dojiwick.domain.errors import ConfigurationError


def assert_binance_ready(*, api_key_env: str, api_secret_env: str, require_live_ack: bool = False) -> tuple[str, str]:
    """Validate Binance API credentials are present in the environment.

    Returns the resolved (api_key, api_secret) tuple on success.
    Raises ConfigurationError if either env var is empty or missing.

    With *require_live_ack*, DOJIWICK_LIVE_ACK=1 must also be set — a
    one-character testnet=false diff plus `make run` must not reach mainnet
    without an explicit second factor.
    """
    api_key = os.environ.get(api_key_env, "").strip()
    api_secret = os.environ.get(api_secret_env, "").strip()
    if not api_key:
        raise ConfigurationError(f"Binance API key not set: env var {api_key_env} is empty or missing")
    if not api_secret:
        raise ConfigurationError(f"Binance API secret not set: env var {api_secret_env} is empty or missing")
    if require_live_ack and os.environ.get("DOJIWICK_LIVE_ACK", "").strip() != "1":
        raise ConfigurationError("testnet=false requires DOJIWICK_LIVE_ACK=1 in the environment (mainnet interlock)")
    return api_key, api_secret
