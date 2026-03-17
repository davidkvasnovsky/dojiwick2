"""Dojiwick CLI entrypoint — dispatches to subcommands."""

from __future__ import annotations

import sys

COMMANDS = {
    "run": "Start the live tick loop",
    "backtest": "Run a backtest on historical data",
    "optimize": "Optuna hyperparameter search",
    "gate": "Research gate evaluation",
    "validate": "Walk-forward / cross-validation",
    "explain": "Debug scope/risk resolution for a pair/regime",
}


def _print_help() -> None:
    print("usage: dojiwick <command> [options]\n")
    print("Batch-first deterministic trading engine.\n")
    print("commands:")
    for name, desc in COMMANDS.items():
        print(f"  {name:<12} {desc}")
    print("\nRun 'dojiwick <command> --help' for command-specific options.")


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        sys.exit(0 if args else 1)

    command, rest = args[0], args[1:]

    if command not in COMMANDS:
        print(f"dojiwick: unknown command '{command}'")
        _print_help()
        sys.exit(1)

    # Rewrite sys.argv so each module's argparse sees the right prog name.
    sys.argv = [f"dojiwick {command}", *rest]

    if command == "run":
        from dojiwick.interfaces.cli.runner import main as run_main

        raise SystemExit(run_main(rest))

    if command == "backtest":
        from dojiwick.interfaces.cli.backtest import main as backtest_main

        backtest_main()

    elif command == "optimize":
        from dojiwick.interfaces.cli.optimize import main as optimize_main

        optimize_main()

    elif command == "gate":
        from dojiwick.interfaces.cli.gate import main as gate_main

        gate_main()

    elif command == "validate":
        from dojiwick.interfaces.cli.validate import main as validate_main

        validate_main()

    elif command == "explain":
        from dojiwick.interfaces.cli.config_explain import main as explain_main

        raise SystemExit(explain_main(rest))


if __name__ == "__main__":
    main()
