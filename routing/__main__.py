"""``python -m routing [train|soft-train|baselines|analyze|…] …``"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        from routing.analysis import main_analyze

        main_analyze(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "baselines":
        from routing.baselines import main_baselines

        main_baselines(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "soft-train":
        from routing.soft_train import main_soft_train

        main_soft_train(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        from routing.figures import main as figures_main

        figures_main()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from routing.benchmark import main_benchmark

        main_benchmark(sys.argv[2:])
        return
    from routing.router import main as router_main

    router_main()


if __name__ == "__main__":
    main()
