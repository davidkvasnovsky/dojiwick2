"""Stable configuration fingerprinting for reproducibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import cast

from .schema import Settings

_RESOLVER_VERSION = "strategy_scope_v2+risk_scope_v1"

_INFRA_ONLY_SECTIONS = frozenset({"database", "optimization", "backtest"})
_INFRA_ONLY_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("system", "log_level"),
        ("system", "shutdown_timeout_sec"),
        ("system", "max_ticks"),
    }
)


@dataclass(slots=True, frozen=True, kw_only=True)
class SettingsFingerprint:
    """Canonical settings fingerprint payload."""

    sha256: str
    trading_sha256: str
    canonical_json: str


def fingerprint_settings(settings: Settings) -> SettingsFingerprint:
    """Return deterministic JSON payload, full SHA-256, and trading-only SHA-256."""

    raw_settings: dict[str, object] = settings.model_dump()
    payload = {
        "resolver_version": _RESOLVER_VERSION,
        "settings": _normalize_value(raw_settings),
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    full_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    trading_payload = {
        "resolver_version": _RESOLVER_VERSION,
        "settings": _normalize_value(_strip_infra(raw_settings)),
    }
    trading_json = json.dumps(trading_payload, sort_keys=True, separators=(",", ":"))
    trading_digest = hashlib.sha256(trading_json.encode("utf-8")).hexdigest()

    return SettingsFingerprint(
        sha256=full_digest,
        trading_sha256=trading_digest,
        canonical_json=canonical_json,
    )


def _normalize_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        d = cast(dict[str, object], value)
        normalized: dict[str, object] = {}
        for key, raw in sorted(d.items(), key=lambda item: str(item[0])):
            normalized[str(key)] = _normalize_value(raw)
        return normalized

    if isinstance(value, list | tuple):
        seq = cast(list[object] | tuple[object, ...], value)
        return [_normalize_value(item) for item in seq]

    if isinstance(value, str | int | float | bool) or value is None:
        return value

    raise TypeError(f"unsupported value for fingerprint normalization: {type(value).__name__}")


def _strip_infra(raw: dict[str, object]) -> dict[str, object]:
    """Remove infra-only sections and fields from raw settings dict."""
    result: dict[str, object] = {}
    for section_name, section_value in raw.items():
        if section_name in _INFRA_ONLY_SECTIONS:
            continue
        if isinstance(section_value, dict):
            section_dict = cast(dict[str, object], section_value)
            filtered: dict[str, object] = {}
            for key, val in section_dict.items():
                if (section_name, key) not in _INFRA_ONLY_FIELDS:
                    filtered[key] = val
            result[section_name] = filtered
        else:
            result[section_name] = section_value
    return result
