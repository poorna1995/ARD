"""Gate 3 — D-AAR P0 wrapper (delegates to route_daar --variant p0 --eval-test)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from daar.routing import route_daar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--theta-s", type=float, default=None)
    p.add_argument("--frozen-lambda", type=float, default=None)
    args = p.parse_args()
    print(
        json.dumps(
            route_daar(
                variant="p0",
                models_dir=args.models_dir,
                theta_s=args.theta_s,
                frozen_lambda=args.frozen_lambda,
                eval_test=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
