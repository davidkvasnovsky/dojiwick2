"""Architecture enforcement tests.

Ensures structural invariants:
- No ``datetime.now`` calls outside ``clock.py``.
- No ``Settings()`` zero-arg construction in ``src/``.
- Override fields stay in sync with their Pydantic param counterparts.
"""

import ast
import pathlib
from dataclasses import fields

from dojiwick.config.scope import StrategyOverrideValues
from dojiwick.config.risk_scope import RiskOverrideValues
from dojiwick.domain.models.value_objects.params import RiskParams, StrategyParams

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "dojiwick"


def _py_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(root.rglob("*.py"))


# No datetime.now outside clock.py


def test_no_datetime_now_outside_clock() -> None:
    """Only ``SystemClock`` (in ``clock.py``) may call ``datetime.now``."""
    violations: list[str] = []
    for path in _py_files(SRC_ROOT):
        if path.name == "clock.py":
            continue
        source = path.read_text()
        if "datetime.now" in source:
            violations.append(str(path.relative_to(SRC_ROOT)))
    assert not violations, f"datetime.now found outside clock.py: {violations}"


# No Settings() zero-arg construction in src/


def test_no_settings_zero_arg_in_src() -> None:
    """``Settings()`` must not be constructed with zero args in production code.

    Tests may use ``Settings()`` freely — this rule only applies to ``src/``.
    """
    violations: list[str] = []
    for path in _py_files(SRC_ROOT):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Settings"
                and not node.args
                and not node.keywords
            ):
                violations.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
    assert not violations, f"Settings() zero-arg construction in src/: {violations}"


# Hexagonal boundary: no infrastructure imports in application/domain

_KNOWN_HEX_ALLOWLIST: set[str] = set()


def _check_no_infra_imports(
    root: pathlib.Path,
    *,
    allowlist: set[str] = frozenset(),  # type: ignore[assignment]
) -> list[str]:
    """Return violations where files under *root* import from ``infrastructure``."""
    violations: list[str] = []
    for path in _py_files(root):
        rel = str(path.relative_to(SRC_ROOT))
        if any(rel.endswith(a) for a in allowlist):
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "infrastructure" in node.module:
                violations.append(f"{rel}:{node.lineno}")
    return violations


def test_no_application_imports_from_infrastructure() -> None:
    """Application layer must not import from infrastructure (except allowlisted files)."""
    violations = _check_no_infra_imports(SRC_ROOT / "application", allowlist=_KNOWN_HEX_ALLOWLIST)
    assert not violations, f"application/ imports from infrastructure/: {violations}"


def test_no_domain_imports_from_infrastructure() -> None:
    """Domain layer must not import from infrastructure."""
    violations = _check_no_infra_imports(SRC_ROOT / "domain")
    assert not violations, f"domain/ imports from infrastructure/: {violations}"


# No direct time calls outside clock.py

_TIME_PATTERNS = ("time.time(", "time.monotonic(", "time.monotonic_ns(")


def test_no_direct_time_calls_outside_clock() -> None:
    """Only ``clock.py`` may call ``time.time()``, ``time.monotonic()`` etc."""
    violations: list[str] = []
    for path in _py_files(SRC_ROOT):
        if path.name == "clock.py":
            continue
        source = path.read_text()
        for pattern in _TIME_PATTERNS:
            if pattern in source:
                violations.append(f"{path.relative_to(SRC_ROOT)} ({pattern})")
    assert not violations, f"Direct time calls found outside clock.py: {violations}"


# No binance references in domain or application


def test_strategy_override_fields_match_strategy_params() -> None:
    """StrategyOverrideValues fields must stay in sync with StrategyParams."""
    override_names = {f.name for f in fields(StrategyOverrideValues)}
    param_names = set(StrategyParams.model_fields)
    assert override_names == param_names, f"drift: {override_names.symmetric_difference(param_names)}"


def test_risk_override_fields_match_risk_params() -> None:
    """RiskOverrideValues fields must stay in sync with RiskParams."""
    override_names = {f.name for f in fields(RiskOverrideValues)}
    param_names = set(RiskParams.model_fields)
    assert override_names == param_names, f"drift: {override_names.symmetric_difference(param_names)}"


def test_no_application_imports_from_config() -> None:
    """Application layer must not import from config/ — use PipelineSettings protocol instead."""
    violations: list[str] = []
    for path in _py_files(SRC_ROOT / "application"):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dojiwick.config"):
                violations.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
    assert not violations, f"application/ imports from config/: {violations}"


def test_no_production_banners() -> None:
    """Status: PRODUCTION banners must not appear in source code."""
    violations: list[str] = []
    for path in _py_files(SRC_ROOT):
        if "Status: PRODUCTION" in path.read_text():
            violations.append(str(path.relative_to(SRC_ROOT)))
    assert not violations, f"Status: PRODUCTION banners found: {violations}"


def test_no_binance_refs_in_domain_or_application() -> None:
    """Domain and application layers must not reference ``binance`` (case-insensitive).

    No allowlisted files — all Phase 2/3 violations resolved.
    """
    violations: list[str] = []
    for root in (SRC_ROOT / "domain", SRC_ROOT / "application"):
        for path in _py_files(root):
            rel = str(path.relative_to(SRC_ROOT))
            if any(rel.endswith(a) for a in _KNOWN_HEX_ALLOWLIST):
                continue
            source = path.read_text().lower()
            if "binance" in source:
                violations.append(rel)
    assert not violations, f"binance references in domain/application: {violations}"
