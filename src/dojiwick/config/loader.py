"""Strict TOML settings loader with deterministic scope parsing."""

import logging
import types
from dataclasses import fields
from pathlib import Path
from collections.abc import Callable
from typing import Any, TypeGuard, cast, get_args, get_origin, get_type_hints
import tomllib

from pydantic import BaseModel, ValidationError

from dojiwick.domain.errors import ConfigurationError

from .risk_scope import RISK_FIELDS, RiskOverrideValues, RiskScopeResolver, RiskScopeRule
from .schema import Settings
from .scope import (
    ScopeSelector,
    StrategyOverrideValues,
    StrategyScopeResolver,
    StrategyScopeRule,
    STRATEGY_FIELDS,
    parse_regime,
)

log = logging.getLogger(__name__)

_KNOWN_SECTIONS = frozenset(
    {
        "system",
        "trading",
        "regime",
        "strategy",
        "risk",
        "ai",
        "backtest",
        "database",
        "optimization",
        "exchange",
        "universe",
        "adaptive",
        "research",
        "flags",
        "scope",
    }
)
_REQUIRED_SECTIONS = (
    "system",
    "trading",
    "regime",
    "strategy",
    "risk",
    "ai",
    "backtest",
    "exchange",
    "optimization",
    "universe",
    "adaptive",
    "research",
)
_SCOPE_STRATEGY_KEYS = frozenset(STRATEGY_FIELDS) | {"id", "priority", "pair", "regime", "strategy"}
_SCOPE_RISK_KEYS = frozenset(RISK_FIELDS) | {"id", "priority", "pair", "regime"}


def _validate_sections(raw: dict[str, object]) -> None:
    unknown = set(raw.keys()) - _KNOWN_SECTIONS
    if unknown:
        raise ConfigurationError(f"unknown config sections: {', '.join(sorted(unknown))}")

    missing = [name for name in _REQUIRED_SECTIONS if name not in raw]
    if missing:
        raise ConfigurationError(f"missing required config sections: {', '.join(missing)}")


_INFRA_ONLY_SECTIONS = frozenset({"database", "flags"})
_INFRA_ONLY_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("system", "log_level"),
        ("system", "shutdown_timeout_sec"),
        ("system", "max_ticks"),
        ("system", "reconciliation_interval_ticks"),
    }
)
_ENFORCE_SECTIONS = _KNOWN_SECTIONS - {"scope"} - _INFRA_ONLY_SECTIONS


def _enforce_explicit_config(raw: dict[str, object], settings: Settings) -> None:
    """Reject missing behavior-bearing fields, log missing infra-only fields.

    Fields with ``None`` default are optional sentinels (e.g., ``sampler_seed: int | None``).
    Fields in ``_INFRA_ONLY_FIELDS`` are infrastructure-only and keep code defaults.
    """
    missing: list[str] = []
    for section_name in _ENFORCE_SECTIONS:
        raw_section = raw.get(section_name)
        toml_keys = raw_section if _is_mapping(raw_section) else {}
        sub_model: object = getattr(settings, section_name)
        if not isinstance(sub_model, BaseModel):
            continue
        model_fields = type(sub_model).model_fields
        for field_name, field_info in model_fields.items():
            if field_name in toml_keys:
                continue
            if (section_name, field_name) in _INFRA_ONLY_FIELDS:
                value = getattr(sub_model, field_name)
                log.debug(
                    "%s.%s not in TOML — using code default: %r",
                    section_name,
                    field_name,
                    value,
                )
                continue
            # Skip optional sentinel fields (default is None or empty collection)
            default = field_info.default
            if default is None:
                continue
            if isinstance(default, tuple) and not default:
                continue
            missing.append(f"{section_name}.{field_name}")
    if missing:
        raise ConfigurationError(
            f"behavior-bearing fields missing from config.toml (no code defaults allowed): {', '.join(missing)}"
        )


def load_settings(path: Path) -> Settings:
    """Load settings from TOML path with strict required-key validation."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    if not _is_mapping(raw):
        raise ConfigurationError("top-level TOML document must be a table")

    _validate_sections(raw)

    # Reject authored active_pairs / primary_pair — they are derived from targets
    trading_section = raw.get("trading")
    if _is_mapping(trading_section):
        if "active_pairs" in trading_section:
            raise ConfigurationError(
                "trading.active_pairs must not be in config — it is derived from [[universe.targets]].display_pair"
            )
        if "primary_pair" in trading_section:
            raise ConfigurationError(
                "trading.primary_pair must not be in config — it is derived from [[universe.targets]].display_pair"
            )

    # Derive active_pairs from targets BEFORE scope parsing
    universe_section = raw.get("universe")
    if not _is_mapping(universe_section):
        raise ConfigurationError("missing required config section: [universe]")
    raw_targets = universe_section.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ConfigurationError("universe.targets must be a non-empty array of [[universe.targets]] tables")
    derived_pairs = tuple(
        str(cast(dict[str, object], t).get("display_pair", "")) for t in cast(list[object], raw_targets)
    )

    # Inject derived trading fields before model_validate
    if not _is_mapping(trading_section):
        raise ConfigurationError("missing required config section: [trading]")
    trading_section["active_pairs"] = list(derived_pairs)
    trading_section["primary_pair"] = derived_pairs[0]

    # Scope parsing uses target-derived pairs
    strategy_scope = _parse_strategy_scope(raw, active_pairs=derived_pairs)
    risk_scope = _parse_risk_scope(raw, active_pairs=derived_pairs)
    raw.pop("scope", None)
    raw["strategy_scope"] = strategy_scope
    raw["risk_scope"] = risk_scope

    try:
        settings = Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc

    _enforce_explicit_config(raw, settings)
    return settings


def _parse_strategy_scope(raw: dict[str, object], *, active_pairs: tuple[str, ...]) -> StrategyScopeResolver:
    """Parse [[scope.strategy]] rules into StrategyScopeResolver."""

    scope_section = _optional_section(raw, "scope")
    if scope_section is None:
        return StrategyScopeResolver.empty()

    unknown_scope_keys = set(scope_section.keys()) - {"strategy", "risk"}
    if unknown_scope_keys:
        raise ConfigurationError(f"unknown keys in [scope]: {', '.join(sorted(unknown_scope_keys))}")

    strategy_entries = scope_section.get("strategy")
    if strategy_entries is None:
        return StrategyScopeResolver.empty()
    if not isinstance(strategy_entries, list):
        raise ConfigurationError("scope.strategy must be an array of tables ([[scope.strategy]])")

    rules: list[StrategyScopeRule] = []
    for index, entry in enumerate(cast(list[object], strategy_entries)):
        validated = _validate_scope_entry(entry, index, section="scope.strategy", allowed_keys=_SCOPE_STRATEGY_KEYS)
        rules.append(_build_scope_rule(validated, index, active_pairs))

    try:
        return StrategyScopeResolver(rules=tuple(rules))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _parse_risk_scope(raw: dict[str, object], *, active_pairs: tuple[str, ...]) -> RiskScopeResolver:
    """Parse [[scope.risk]] rules into RiskScopeResolver."""

    scope_section = _optional_section(raw, "scope")
    if scope_section is None:
        return RiskScopeResolver.empty()

    risk_entries = scope_section.get("risk")
    if risk_entries is None:
        return RiskScopeResolver.empty()
    if not isinstance(risk_entries, list):
        raise ConfigurationError("scope.risk must be an array of tables ([[scope.risk]])")

    rules: list[RiskScopeRule] = []
    for index, entry in enumerate(cast(list[object], risk_entries)):
        validated = _validate_scope_entry(entry, index, section="scope.risk", allowed_keys=_SCOPE_RISK_KEYS)
        rules.append(_build_risk_scope_rule(validated, index, active_pairs))

    try:
        return RiskScopeResolver(rules=tuple(rules))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _validate_scope_entry(
    entry: object, index: int, *, section: str, allowed_keys: frozenset[str]
) -> dict[str, object]:
    """Validate a single scope entry structure and return typed dict."""
    if not _is_mapping(entry):
        raise ConfigurationError(f"{section}[{index}] must be a TOML table")

    unknown_keys = set(entry.keys()) - allowed_keys
    if unknown_keys:
        raise ConfigurationError(f"unknown keys in {section}[{index}]: {', '.join(sorted(unknown_keys))}")
    return entry


def _parse_scope_header(
    entry: dict[str, object], index: int, *, section: str, active_pairs: tuple[str, ...]
) -> tuple[str, int, ScopeSelector]:
    """Parse and validate the common id/priority/pair/regime header of a scope entry."""
    rule_id = entry.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ConfigurationError(f"{section}[{index}].id must be a non-empty string")

    priority = entry.get("priority")
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ConfigurationError(f"{section}[{index}].priority must be an integer")

    pair = entry.get("pair")
    if pair is not None:
        if not isinstance(pair, str) or not pair:
            raise ConfigurationError(f"{section}[{index}].pair must be a non-empty string")
        if pair not in active_pairs:
            raise ConfigurationError(f"{section}[{index}].pair must be one of active_pairs: {pair}")

    regime = entry.get("regime")
    parsed_regime = None
    if regime is not None:
        if not isinstance(regime, str):
            raise ConfigurationError(f"{section}[{index}].regime must be a string")
        try:
            parsed_regime = parse_regime(regime)
        except ValueError as exc:
            raise ConfigurationError(f"{section}[{index}].regime: {exc}") from exc

    strategy_name = entry.get("strategy")
    if strategy_name is not None:
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ConfigurationError(f"{section}[{index}].strategy must be a non-empty string")

    return rule_id, priority, ScopeSelector(pair=pair, regime=parsed_regime, strategy=strategy_name)


def _extract_override_values(
    cls: type,
    entry: dict[str, object],
    index: int,
    section: str,
) -> dict[str, Any]:
    """Extract override values from a scope entry using dataclass field metadata."""
    type_parsers: dict[type, Callable[[object, str], object]] = {
        float: _optional_float,
        int: _optional_int,
        str: _optional_str,
        bool: _optional_bool,
    }
    hints = get_type_hints(cls)
    result: dict[str, Any] = {}
    for f in fields(cls):
        annotation = hints[f.name]
        # Unwrap X | None → X
        origin = get_origin(annotation)
        base_type: type | None = None
        if origin is types.UnionType:
            args = get_args(annotation)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                base_type = non_none[0]
        elif isinstance(annotation, type):
            base_type = annotation
        parser = type_parsers.get(base_type) if base_type is not None else None
        if parser is not None:
            label = f"{section}[{index}].{f.name}"
            result[f.name] = parser(entry.get(f.name), label)
    return result


def _build_scope_rule(entry: dict[str, object], index: int, active_pairs: tuple[str, ...]) -> StrategyScopeRule:
    """Parse and construct a single StrategyScopeRule from a validated entry."""
    rule_id, priority, selector = _parse_scope_header(entry, index, section="scope.strategy", active_pairs=active_pairs)

    values = StrategyOverrideValues(**_extract_override_values(StrategyOverrideValues, entry, index, "scope.strategy"))

    try:
        return StrategyScopeRule(
            id=rule_id,
            priority=priority,
            selector=selector,
            values=values,
        )
    except ValueError as exc:
        raise ConfigurationError(f"scope.strategy[{index}]: {exc}") from exc


def _build_risk_scope_rule(entry: dict[str, object], index: int, active_pairs: tuple[str, ...]) -> RiskScopeRule:
    """Parse and construct a single RiskScopeRule from a validated entry."""
    rule_id, priority, selector = _parse_scope_header(entry, index, section="scope.risk", active_pairs=active_pairs)

    values = RiskOverrideValues(**_extract_override_values(RiskOverrideValues, entry, index, "scope.risk"))

    try:
        return RiskScopeRule(
            id=rule_id,
            priority=priority,
            selector=selector,
            values=values,
        )
    except ValueError as exc:
        raise ConfigurationError(f"scope.risk[{index}]: {exc}") from exc


def _optional_section(raw: dict[str, object], name: str) -> dict[str, object] | None:
    value = raw.get(name)
    if value is None:
        return None
    if not _is_mapping(value):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    return dict(value)


def _optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a number")
    return float(value)


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{field} must be an integer")
    return value


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a boolean")
    return value


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{field} must be a string")
    return value


def _is_mapping(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast(dict[object, object], value))
