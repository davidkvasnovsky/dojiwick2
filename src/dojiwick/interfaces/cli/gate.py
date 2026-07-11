"""CLI entrypoint for research gate evaluation.

Usage::

    python -m dojiwick.interfaces.cli.gate \
        --config config.toml --start 2025-01-01 --end 2025-06-01 \
        --params-json '{"stop_atr_mult": 2.0, "rr_ratio": 2.5, ...}'
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research gate evaluation on a parameter set.")
    from dojiwick.interfaces.cli._shared import add_common_args

    add_common_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--params-file", type=Path, help="JSON file with parameter set")
    group.add_argument("--params-json", type=str, help="Inline JSON parameter set")
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()

    from dojiwick.config.param_tuning import apply_params
    from dojiwick.interfaces.cli._shared import build_gate_evaluator, load_settings_and_series

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
    else:
        params = json.loads(args.params_json)

    settings, series, cleanup = await load_settings_and_series(args)

    try:
        tuned = apply_params(settings, params)
        evaluator = build_gate_evaluator(tuned, series, apply_tuned_from=settings)

        log.info("running research gate evaluation")
        result = await evaluator.evaluate(params)

        from dojiwick.interfaces.cli._shared import print_gate_block

        return print_gate_block(result)

    finally:
        await cleanup()


def main() -> int:
    from dojiwick.interfaces.cli._shared import setup_env

    setup_env()
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
