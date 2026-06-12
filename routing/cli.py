"""
Unified CLI — ``research-route`` / ``python -m routing``.

Default subcommand: ``verify``.
"""

from __future__ import annotations

import argparse
import sys


def _dispatch(argv: list[str]) -> None:
    cmd, rest = argv[0], argv[1:]
    if cmd == "analyze":
        from research.analysis import main_analyze

        main_analyze(rest)
    elif cmd == "baselines":
        from research.baselines import main_baselines

        main_baselines(rest)
    elif cmd == "soft-train":
        from research.soft_train import main_soft_train

        main_soft_train(rest)
    elif cmd == "figures":
        from research.figures import main as figures_main

        figures_main()
    elif cmd == "benchmark":
        from eval.benchmark import main_benchmark

        main_benchmark(rest)
    elif cmd in ("score-routes", "score"):
        from eval.score import main_score_routes

        main_score_routes(rest)
    elif cmd == "verify":
        from routing.verify import main_verify

        main_verify(rest)
    elif cmd == "oracle-bounds":
        from eval.oracle_bounds import main_oracle_bounds

        main_oracle_bounds(rest)
    else:
        from router.experiments import main as router_main

        router_main(argv)


def _print_top_level_help() -> None:
    parser = argparse.ArgumentParser(
        prog="research-route",
        description="QCE router CLI (train, eval, analyze, benchmark, verify, …).",
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        help=(
            "verify (default) | soft-train | train | eval | tune | analyze | baselines | "
            "benchmark | score-routes | figures | oracle-bounds | …"
        ),
    )
    parser.print_help()


def main(argv: list[str] | None = None) -> None:
    from config.global_config.runtime import configure_logging

    configure_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from routing.verify import main_verify

        main_verify([])
        return
    if argv[0] in ("-h", "--help") and len(argv) == 1:
        _print_top_level_help()
        return
    _dispatch(argv)


if __name__ == "__main__":
    main()
