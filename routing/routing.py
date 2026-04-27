from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FEATURES_PATH = (
    REPO_ROOT / "results1" / "unified_baseline" / "gaia" / "features_complexity.parquet"
)


class Routing:
    """
    Route a query to the right strategy from complexity band.

    Band mapping:
      - LLM -> vanilla baseline
      - CoT -> zero-shot CoT baseline
      - SA  -> ReAct agent
      - MA  -> multi-agent (placeholder)
    """


    def __init__(
        self,
        query: str,
        modality: str | None = None,
        features_path: Path | None = None,
    ) -> None:
        self.query = query
        self.features_path = features_path or DEFAULT_FEATURES_PATH
        self.modality = modality or self._lookup_complexity_band(query)

    def _lookup_complexity_band(self, query: str) -> str:
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        df = pd.read_parquet(self.features_path)
        if "query" not in df.columns or "complexity_band" not in df.columns:
            raise ValueError(
                "Expected columns ['query', 'complexity_band'] in features parquet."
            )

        matches = df.loc[df["query"] == query, "complexity_band"]
        if matches.empty:
            raise ValueError("Query not found in features file.")
        return str(matches.iloc[0])

    def route(self):
        band = str(self.modality).upper()

        if band == "LLM":
            from baselines.runner import run_selected

            return {
                "router": "baseline",
                "modality": "vanilla",
                "model": "llama-3.3-70b-versatile",
            }
        if band == "COT":
            from baselines.runner import run_selected, RunSelection, QueryInput
            query_input = QueryInput(
                query_id="7",
                query=self.query,
                dataset="gaia",
                ground_truth="",
            )
            selection = RunSelection(
                datasets=("gaia",),
                modalities=("zero_shot_cot",),
                model="llama-3.3-70b-versatile",
                max_tokens=1024,
                temperature=0.0,
            )
            results = run_selected(
                queries=[query_input],
                selection=selection,
                verbose_prompts=False,
            )
            return {
                "router": "baseline",
                "modality": "zero_shot_cot",
                "model": "llama-3.3-70b-versatile",
                'answer': results[0].predicted_answer,
            }
        if band == "SA":
            from agents.react.react import build_agent
            agent = build_agent()
            answer = agent.run(self.query)
            return {
                "router": "react",
                "modality": "SA",
                "answer": answer,
            }
        if band == "MA":
            return {
                "router": "multi_agent",
                "modality": "MA",
                "message": "MA pipeline not implemented yet.",
            }

        raise ValueError(f"Unsupported complexity band: {band}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route GAIA queries by complexity band.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to features parquet (must include query, complexity_band).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run routing for a single query string.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for batch routing (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for batch routing (exclusive).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.query is not None:
        result = Routing(query=args.query, features_path=args.features_path).route()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if args.start is None or args.end is None:
            raise ValueError("For batch routing, provide both --start and --end.")

        df = pd.read_parquet(args.features_path)
        if "query" not in df.columns:
            raise ValueError("Expected 'query' column in features parquet.")

        for idx, query in df["query"].iloc[args.start:args.end].items():
            result = Routing(query=str(query), features_path=args.features_path).route()
            print(f"\n[{idx}]")
            print(json.dumps(result, indent=2, ensure_ascii=False))