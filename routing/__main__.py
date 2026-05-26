"""``python -m routing [train|eval|tune|ablation|lodo|analyze] …``"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        from routing.analysis import main_analyze

        main_analyze(sys.argv[2:])
        return
    from routing.router import main as router_main

    router_main()


if __name__ == "__main__":
    main()
