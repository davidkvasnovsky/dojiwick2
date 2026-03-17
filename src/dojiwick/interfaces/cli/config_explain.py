"""Explain deterministic strategy-scope and risk-scope resolution for a pair/regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dojiwick.config.fingerprint import fingerprint_settings
from dojiwick.config.loader import load_settings
from dojiwick.config.risk_scope import RiskResolutionTrace, RISK_FIELDS
from dojiwick.config.scope import ResolutionTrace, parse_regime, regime_name
from dojiwick.domain.enums import MarketState


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for config explain command."""

    parser = argparse.ArgumentParser(description="Explain resolved strategy config for one pair/regime context")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--regime")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def _parse_regime_arg(raw: str | None) -> MarketState | None:
    if raw is None:
        return None
    return parse_regime(raw)


def _render_table(trace: ResolutionTrace, risk_trace: RiskResolutionTrace, config_hash: str) -> str:
    lines: list[str] = []
    lines.append(f"config_hash: {config_hash}")
    lines.append(f"pair: {trace.pair}")
    lines.append(f"regime: {regime_name(trace.regime) if trace.regime is not None else '*'}")
    lines.append("")
    lines.append("Matched rules (sorted):")

    if not trace.matched_rules:
        lines.append("- <none>")
    else:
        for rule in trace.matched_rules:
            lines.append(
                f"- {rule.rule_id}: priority={rule.priority} specificity={rule.specificity} selector={rule.selector}"
            )

    lines.append("")
    lines.append("Field winners:")
    if not trace.field_winners:
        lines.append("- <none>")
    else:
        for winner in trace.field_winners:
            lines.append(
                "- "
                f"{winner.field_name}={winner.value} "
                f"from={winner.rule_id} "
                f"(priority={winner.priority} specificity={winner.specificity} selector={winner.selector})"
            )

    lines.append("")
    lines.append("Resolved strategy:")
    lines.append(f"- default_variant={trace.resolved.default_variant}")
    lines.append(f"- stop_atr_mult={trace.resolved.stop_atr_mult}")
    lines.append(f"- rr_ratio={trace.resolved.rr_ratio}")
    lines.append(f"- min_stop_distance_pct={trace.resolved.min_stop_distance_pct}")

    lines.append("")
    lines.append("Resolved risk:")
    for field in RISK_FIELDS:
        lines.append(f"- {field}={getattr(risk_trace.resolved, field)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = load_settings(Path(args.config))
    regime = _parse_regime_arg(args.regime)
    trace = settings.strategy_scope.explain(args.pair, regime, settings.strategy)
    risk_trace = settings.risk_scope.explain(args.pair, regime, settings.risk.params)
    fingerprint = fingerprint_settings(settings)

    if args.format == "json":
        payload = {
            "config_hash": fingerprint.sha256,
            "trace": trace.as_json(),
            "risk_trace": risk_trace.as_json(),
        }
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0

    print(_render_table(trace, risk_trace, fingerprint.sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
