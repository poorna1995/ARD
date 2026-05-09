from __future__ import annotations
import os
import argparse
import json
import sys
from pathlib import Path
from difficulty.feature_measure import TaskComplexityAnalyzer
import pandas as pd

from prompts.prompts import DATASETS, AGENTS

REPO_ROOT = Path(__file__).resolve().parents[1]
# ✅ Simplest - direct string key access
MODELS = {
    "raw":        {"primary": "llama-3.3-70b-versatile", "secondary": "gpt-4o"},
    "cot":        {"primary": "llama-3.3-70b-versatile",                  "secondary": "gpt-4o"},
    "react":      {"primary": "llama-3.3-70b-versatile",                  "secondary": "gpt-4o"},
    "multiagent": {"primary": "llama-3.3-70b-versatile",                  "secondary": "gpt-4o"},
}
AGENT_MODULES = {
    "raw": "agent.raw",
    "cot": "agent.cot",
    "react": "agent.react",
    "multiagent": "agent.multiagent",
}
DATASETS = ["gaia", "mmlu_pro", "math_hard", "swe_bench_verified"]
AGENTS = ['raw', 'cot', 'react', 'multiagent']

class Router:
    # for router i need to have an nput of query,complexity score, threshold, model name, datasetname, agent name,adn other parameters that are needed to route the query to the right agent.
    def __init__(
        self,
        query : str,
        analyzer : TaskComplexityAnalyzer,
        dataset : str,
        **kwargs,
    ):
        self.query = query
        self.analyzer = analyzer
        self.dataset = dataset
        self.other_parameters = kwargs

        analysis = analyzer.analyze(query)
        self.overall = analysis.overall
        self.task_length = analysis.task_length
        self.reasoning_depth = analysis.reasoning_depth
        self.tool_dependency = analysis.tool_dependency
        self.domain_breadth = analysis.domain_breadth
        self.task_type = analysis.task_type

        self.agent = self._assign_agent()
        self.model = MODELS[self.agent]
    def inspect(self) -> dict:
        """Return complexity + routing decision WITHOUT running agent."""
        return {
            "query": self.query,
            "overall": self.overall,
            "task_length": self.task_length,
            "reasoning_depth": self.reasoning_depth,
            "tool_dependency": self.tool_dependency,
            "domain_breadth": self.domain_breadth,
            "task_type": self.task_type,
            "assigned_agent": self.agent,
            "model_primary": self.get_model(),
            "model_secondary": self.get_model(use_secondary=True),
        }

    # def _assign_agent(self) -> str:
    #     score = self.overall
    #     threshold = self.analyzer.thresholds

    #     if score < threshold["raw"] and self.task_length < 7 and self.domain_breadth < 1:
    #         return "raw"
    #     elif score < threshold["cot"] and self.reasoning_depth < 1:
    #         return "cot"
    #     elif score < threshold["react"] and self.tool_dependency < 4:
    #         return "react"
    #     else:
    #         return "multiagent"


    def _assign_agent(self) -> str:
        t  = self.analyzer.thresholds
        ov = t["overall"]

        # Per-feature thresholds only exist after fit(); fall back to raw scores
        tl = t.get("task_length",    {"raw": 3.0, "cot": 5.0, "react": 7.0})
        rd = t.get("reasoning_depth",{"raw": 3.0, "cot": 5.0, "react": 7.0})
        td = t.get("tool_dependency",{"raw": 0.0, "cot": 1.0, "react": 4.0})
        db = t.get("domain_breadth", {"raw": 0.0, "cot": 2.0, "react": 4.0})

        if (self.overall        < ov["raw"]
                and self.task_length    < tl["raw"]):
            return "raw"

        if (self.overall         < ov["cot"]
                and self.reasoning_depth < rd["cot"]):
            return "cot"

        if (self.overall         < ov["react"]
                and self.tool_dependency < td["react"]):
            return "react"

        return "multiagent"


    def get_model(self, use_secondary: bool = False) -> str:
        key = 'secondary' if use_secondary else 'primary'
        return self.model[key]


    # route the query to the right agent
    def run(self, expected_answer: str | None = None) -> dict:
        from importlib import import_module

        agent_module = import_module(AGENT_MODULES[self.agent])
        agent_fn = agent_module.run
        response = agent_fn(
            query   = self.query,
            model   = self.get_model(),
            dataset = self.dataset,
            expected_answer  = expected_answer,
            **self.other_parameters,
        )
        return {
            "query":            self.query,
            "agent":            self.agent,
            "model":            self.get_model(),
            "overall":          self.overall,
            "response":         response,



        }

from collections import Counter

def evaluate_routing(df, analyzer):
    counts = Counter()
    total = len(df)

    for query in df["query"]:
        router = Router(
            query=query,
            analyzer=analyzer,
            dataset="gaia"
        )
        result = router.inspect()   # ✅ no agent execution
        counts[result["assigned_agent"]] += 1

    # Convert to percentages
    stats = []
    for agent in AGENTS:
        count = counts[agent]
        percentage = (count / total) * 100 if total > 0 else 0
        stats.append({
            "agent": agent,
            "count": count,
            "percentage": round(percentage, 2)
        })

    return stats

if __name__ == "__main__":
    df = pd.read_parquet("datasets/golden/gaia.parquet")

    analyzer = TaskComplexityAnalyzer()
    analyzer.fit(df["query"].tolist())

    # Route only (no agent execution); collect all routing values.
    results = []
    for _, row in df.iterrows():
        router = Router(
            query=row["query"],
            analyzer=analyzer,
            dataset="gaia",
        )
        results.append(router.inspect())

    stats = evaluate_routing(df, analyzer)

    print("\nRouting Distribution:\n")
    for s in stats:
        print(f"{s['agent']:10} -> {s['count']:5} ({s['percentage']:6.2f}%)")





    # ── Step 4: Build DataFrame ────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    # ── Step 5: Summary ────────────────────────────────────────────────────
    print("\nRouting Distribution:")
    print(results_df["assigned_agent"].value_counts())

    print("\nAverage Complexity per Agent:")
    print(results_df.groupby("assigned_agent")["overall"].mean().round(3))

    # ── Step 6: Save ───────────────────────────────────────────────────────
    output_path = REPO_ROOT / "datasets" / "complexity" / "gaia_routed1.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    print(f"\n✅ Saved → {output_path}")