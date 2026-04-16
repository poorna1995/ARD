# import time
# import json
# import csv
# import os
# from dataclasses import dataclass, field, asdict
# from typing import Optional
# from openai import OpenAI  # works for GPT; swap client for QWEN/Llama
# from dotenv import load_dotenv
# load_dotenv()
# import pandas as pd
# # ─────────────────────────────────────────────
# # 1. CONFIG
# # ─────────────────────────────────────────────
# LLM_CONFIG = {
#     "model": "gpt-4o-mini",
#     "max_tokens": 1024,
#     "temperature": 0.1,
# }

# COST_PER_1K_TOKENS = {
#     "gpt-4o-mini":      {"input": 0.00015, "output": 0.0006},
#     "qwen-2-72b":       {"input": 0.0009,  "output": 0.0009},
#     "gemini-1.5-flash": {"input": 0.000075,"output": 0.0003},
#     "llama-3.1-70b":    {"input": 0.00059, "output": 0.00079},
# }

# PATHS = {
#     "gaia":       "../datasets/processed/processed_gaia.parquet",
#     # "swe_bench":  "../datasets/processed/processed_swe_bench.parquet",
#     # "math_hard":  "../datasets/processed/processed_math_hard.parquet",
#     # "agentbench": "../datasets/processed/processed_agentbench.parquet",
# }
# # ─────────────────────────────────────────────
# # 2. DATA STRUCTURES
# # ─────────────────────────────────────────────

# MODELS = ["gpt-4o-mini", "gemini-1.5-flash", "llama-3.1-70b"]
# @dataclass
# class Query:
#     id: str
#     dataset: str          # GAIA | SWE-Bench | Math-Hard | AgentBench
#     question: str
#     ground_truth: str
#     metadata: dict = field(default_factory=dict)

# @dataclass
# class Result:
#     query_id: str
#     dataset: str
#     question: str
#     ground_truth: str
#     predicted: str
#     is_correct: bool
#     input_tokens: int
#     output_tokens: int
#     total_tokens: int
#     cost_usd: float
#     time_seconds: float
#     model: str
#     error: Optional[str] = None



# class Vanilla():
#     def __init__(self, model_name: str):
#         if model_name not in MODELS:
#             raise ValueError(f"Model '{model_name}' not supported. Choose from: {MODELS}")
#         self.model_name = model_name
#         self.client = OpenAI()  # swap for QWEN/Llama client if needed
        
    
#     def _load_gaia(self, path: str) -> list[Query]:
#         df = pd.read_parquet(path)

#         print(f"[GAIA] Loaded {len(df)} rows")
#         print(f"[GAIA] Columns: {df.columns.tolist()}")   # inspect once, then lock columns below

#         queries = []
#         for _, row in df.iterrows():
#             queries.append(Query(
#                 id           = str(row.get("id",   row.name)),   # fallback to index
#                 dataset      = "GAIA",
#                 question     = str(row["query"]),                   # ← adjust col name if needed
#                 ground_truth = str(row["answer"]),               # ← adjust col name if needed
#                 metadata     = {
#                     "level": row.get("level", None),
#                     "file":  row.get("file_name", None),
#                 }
#             ))
#         return queries
    
#     def _load_unified_dataset(self) -> list[Query]:
#         queries = []
#         queries += self._load_gaia(PATHS["gaia"])

#         # Uncomment as you add more processed datasets:
#         # queries += self._load_swe_bench(PATHS["swe_bench"])
#         # queries += self._load_math_hard(PATHS["math_hard"])
#         # queries += self._load_agentbench(PATHS["agentbench"])

#         print(f"\nTotal queries loaded: {len(queries)}")
#         return queries

#     def _vanilla(self, query: Query) -> Result:
#         model = LLM_CONFIG["model"]
#         start = time.perf_counter()
#         error = None
#         predicted = ""
#         usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=model,
#                 max_tokens=LLM_CONFIG["max_tokens"],
#                 temperature=LLM_CONFIG["temperature"],
#                 messages=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user",   "content": query.question},
#                 ],
#             )
#             predicted = response.choices[0].message.content.strip()
#             usage = {
#                 "prompt_tokens":     response.usage.prompt_tokens,
#                 "completion_tokens": response.usage.completion_tokens,
#                 "total_tokens":      response.usage.total_tokens,
#             }

#         except Exception as e:
#             error = str(e)
#             error_type = type(e).__name__

#         elapsed = time.perf_counter() - start

#         rates = COST_PER_1K_TOKENS.get(model, {"input": 0, "output": 0})
#         cost  = (
#             usage["prompt_tokens"]     / 1000 * rates["input"] +
#             usage["completion_tokens"] / 1000 * rates["output"]
#         )

#         is_correct = predicted.strip().lower() == query.ground_truth.strip().lower()

#         return Result(
#             query_id     = query.id,
#             dataset      = query.dataset,
#             question     = query.question,
#             ground_truth = query.ground_truth,
#             predicted    = predicted,
#             is_correct   = is_correct,
#             input_tokens = usage["prompt_tokens"],
#             output_tokens= usage["completion_tokens"],
#             total_tokens = usage["total_tokens"],
#             cost_usd     = round(cost, 6),
#             time_seconds = round(elapsed, 3),
#             model        = model,
#             level        = query.metadata.get("level"),
#             error        = error,
#         )
    
    
    
#     def _run_baseline(self, queries: list[Query]) -> list[Result]:
#         results = []
#         for i, q in enumerate(queries):
#             print(f"[{i+1}/{len(queries)}] {q.dataset} | {q.id}")
#             print(f"   ❓ Question : {q.question[:100]}...")   # trim long questions
#             print(f"   ✅ Expected : {q.ground_truth}")

#             r = self._vanilla(q)
#             results.append(r)

#             status = "✓" if r.is_correct else "✗"
#             print(f"   🤖 Predicted: {r.predicted}")
#             print(f"   {status} correct={r.is_correct}  tokens={r.total_tokens}  "
#                 f"cost=${r.cost_usd:.5f}  time={r.time_seconds}s")
#             if r.error:
#                 print(f"   ⚠ error: {r.error} ({r.error_type})")
#             print("-" * 60)

#         return results



#     def _evaluate(self, results: list[Result]):
#         total   = len(results)
#         correct = sum(r.is_correct for r in results)

#         print("\n─── BASELINE SUMMARY ───────────────────────────")
#         print(f"  Model         : {results[0].model}")
#         print(f"  Total queries : {total}")
#         print(f"  Accuracy      : {correct}/{total} = {correct/total:.1%}")
#         print(f"  Total tokens  : {sum(r.total_tokens  for r in results):,}")
#         print(f"  Total cost    : ${sum(r.cost_usd     for r in results):.4f}")
#         print(f"  Avg latency   : {sum(r.time_seconds  for r in results)/total:.2f}s")
#         print(f"  Errors        : {sum(1 for r in results if r.error)}")

#         # Per-dataset breakdown
#         print("\n─── PER DATASET ─────────────────────────────────")
#         for ds in sorted({r.dataset for r in results}):
#             ds_res = [r for r in results if r.dataset == ds]
#             acc = sum(r.is_correct for r in ds_res) / len(ds_res)
#             print(f"  [{ds}]  accuracy={acc:.1%}  n={len(ds_res)}"
#                 f"  cost=${sum(r.cost_usd for r in ds_res):.4f}")

#         # GAIA level breakdown (if metadata present)
#         gaia_res = [r for r in results if r.dataset == "GAIA"]
#         if gaia_res:
#             df_eval = pd.DataFrame([asdict(r) for r in gaia_res])
#             if "level" in df_eval.columns:
#                 print("\n─── GAIA BY LEVEL ───────────────────────────────")
#                 # print(df_eval.groupby("level")["is_correct"].mean().to_string())
#                 print(
#                 (df_eval.groupby("level")["is_correct"].mean() * 100)
#                 .round(2)
#                 .astype(str) + "%"
#     )


#     def _save_results(results: list[Result]):
#         model_tag = LLM_CONFIG["model"].replace("/", "-")
        
#         # Save path: datasets/baseline/gaia/v1_{model_name}.parquet
#         save_dir = os.path.join("..", "datasets", "baseline", "gaia")
#         os.makedirs(save_dir, exist_ok=True)
        
#         parquet_path = os.path.join(save_dir, f"v1_{model_tag}.parquet")
        
#         pd.DataFrame([asdict(r) for r in results]).to_parquet(parquet_path, index=False)
        
#         print(f"\n✅ Saved → {parquet_path}")
#         print(f"   Rows    : {len(results)}")
#         print(f"   Model   : {model_tag}")
#         print(f"   Correct : {sum(r.is_correct for r in results)}/{len(results)}")
        
#     def forward(self):
#         queries = self._load_unified_dataset()
#         results = self._run_baseline(queries)
#         self._evaluate(results)
#         self._save_results(results)
    
    
    
    
    
    
    

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from agents.vanallia_operator import vanallia_operator

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    """You are an expert question answering system evaluated on the GAIA benchmark.

## Output Rules
- Return ONLY the final answer — nothing else
- Answers are always one of: a single word, a number, a short phrase, or a name
- No explanations, no sentences, no punctuation at the end
- No preamble like "The answer is..." or "Based on..."
- If the answer is a number, return just a single number after one comma dont return (e.g. 42, 3.14)
- If the answer is a name, return just the name (e.g. Paris, Einstein)
- If the answer is a word, return just that word (e.g. egalitarian, blue)
- If the answer is a short phrase, return just that phrase (e.g. "New York", "machine learning")

## Now answer the following question with ONLY the final answer:
"""
)

COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini":      {"input": 0.00015,  "output": 0.0006},
    "gpt-4o":           {"input": 0.005,    "output": 0.015},
    "qwen-2-72b":       {"input": 0.0009,   "output": 0.0009},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "llama-3.1-70b":    {"input": 0.00059,  "output": 0.00079},
    "llama-3.1-8b":     {"input": 0.00018,  "output": 0.00018},
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    One entry per model. Pass base_url for non-OpenAI endpoints
    (Ollama, vLLM, Together, Groq, etc.).
    """
    name: str
    max_tokens: int = 1024
    temperature: float = 0.
    base_url: Optional[str] = None   # e.g. "http://localhost:11434/v1" for Ollama
    api_key: Optional[str] = None    # leave None → reads OPENAI_API_KEY from env


@dataclass
class Query:
    id: str
    dataset: str
    question: str
    ground_truth: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Result:
    query_id: str
    dataset: str
    model: str
    question: str
    ground_truth: str
    predicted: str
    is_correct: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    time_seconds: float
    error: Optional[str] = None
    # Flattened metadata — extend as needed
    level: Optional[str] = None
    category: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER  (registry pattern — add loaders without touching other classes)
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Registry-based dataset loader.

    Built-in loaders: gaia, swe_bench, math_hard, agentbench
    Custom loaders:   call loader.register("my_ds", my_fn) before pipeline.run()

    Loader signature:
        def my_fn(path: str) -> list[Query]: ...
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[str], list[Query]]] = {}
        # ── Register all built-in loaders ──────────────────────────────────
        self.register("gaia",       self._load_gaia)
        self.register("swe_bench",  self._load_swe_bench)
        self.register("math_hard",  self._load_math_hard)
        self.register("agentbench", self._load_agentbench)

    # ── Public API ────────────────────────────────────────────────────────

    def register(self, name: str, fn: Callable[[str], list[Query]]) -> None:
        self._registry[name] = fn

    def load(self, dataset_name: str, path: str) -> list[Query]:
        if dataset_name not in self._registry:
            raise ValueError(
                f"No loader registered for '{dataset_name}'. "
                f"Available: {list(self._registry)}"
            )
        queries = self._registry[dataset_name](path)
        print(f"  [DataLoader] '{dataset_name}' → {len(queries)} queries  ({path})")
        return queries

    # ── Built-in loaders ─────────────────────────────────────────────────

    def _load_gaia(self, path: str) -> list[Query]:
        df = pd.read_parquet(path)
        return [
            Query(
                id=str(row.get("id", idx)),
                dataset="GAIA",
                question=str(row["query"]),
                ground_truth=str(row["answer"]),
                metadata={
                    "level": row.get("level"),
                    "file":  row.get("file_name"),
                },
            )
            for idx, row in df.iterrows()
        ]

    def _load_swe_bench(self, path: str) -> list[Query]:
        df = pd.read_parquet(path)
        return [
            Query(
                id=str(row.get("instance_id", idx)),
                dataset="SWE-Bench",
                question=str(row["problem_statement"]),
                ground_truth=str(row.get("patch", "")),
                metadata={"repo": row.get("repo"), "level": row.get("difficulty")},
            )
            for idx, row in df.iterrows()
        ]

    def _load_math_hard(self, path: str) -> list[Query]:
        df = pd.read_parquet(path)
        return [
            Query(
                id=str(row.get("id", idx)),
                dataset="Math-Hard",
                question=str(row["problem"]),
                ground_truth=str(row["solution"]),
                metadata={"level": row.get("level"), "category": row.get("type")},
            )
            for idx, row in df.iterrows()
        ]

    def _load_agentbench(self, path: str) -> list[Query]:
        df = pd.read_parquet(path)
        return [
            Query(
                id=str(row.get("id", idx)),
                dataset="AgentBench",
                question=str(row["instruction"]),
                ground_truth=str(row["answer"]),
                metadata={"category": row.get("category"), "level": row.get("level")},
            )
            for idx, row in df.iterrows()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT  (OpenAI-compatible; swap base_url for other providers)
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around any OpenAI-compatible endpoint.
    One instance per ModelConfig — created fresh inside the pipeline loop.
    """

    def __init__(self, config: ModelConfig, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.config = config
        self.system_prompt = system_prompt
        kwargs: dict = {}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key
        self._client = OpenAI(**kwargs)

    def generate(self, query: Query) -> Result:
        cfg = self.config
        predicted, usage, error, elapsed = vanallia_operator(
            client=self._client,
            model_name=cfg.name,
            question=query.question,
            system_prompt=self.system_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        rates   = COST_PER_1K_TOKENS.get(cfg.name, {"input": 0.0, "output": 0.0})
        cost    = (
            usage["prompt_tokens"]     / 1000 * rates["input"] +
            usage["completion_tokens"] / 1000 * rates["output"]
        )
        is_correct = predicted.strip().lower() == query.ground_truth.strip().lower()

        return Result(
            query_id=query.id,
            dataset=query.dataset,
            model=cfg.name,
            question=query.question,
            ground_truth=query.ground_truth,
            predicted=predicted,
            is_correct=is_correct,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cost_usd=round(cost, 6),
            time_seconds=round(elapsed, 3),
            error=error,
            level=query.metadata.get("level"),
            category=query.metadata.get("category"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Computes a rich metrics dict from a list[Result].
    Pure function — no I/O, no state.
    """

    def compute(self, results: list[Result]) -> dict:
        if not results:
            return {}

        total   = len(results)
        correct = sum(r.is_correct for r in results)
        df      = pd.DataFrame([asdict(r) for r in results])

        metrics: dict = {
            "model":           results[0].model,
            "dataset":         results[0].dataset,
            "total":           total,
            "correct":         correct,
            "accuracy":        round(correct / total, 4),
            "accuracy_pct":    f"{correct / total:.1%}",
            "total_tokens":    int(df["total_tokens"].sum()),
            "avg_input_tokens":  round(df["input_tokens"].mean(), 1),
            "avg_output_tokens": round(df["output_tokens"].mean(), 1),
            "total_cost_usd":  round(df["cost_usd"].sum(), 4),
            "avg_latency_s":   round(df["time_seconds"].mean(), 3),
            "p50_latency_s":   round(df["time_seconds"].median(), 3),
            "p95_latency_s":   round(df["time_seconds"].quantile(0.95), 3),
            "errors":          int(df["error"].notna().sum()),
            "error_rate":      round(df["error"].notna().mean(), 4),
        }

        # ── Optional breakdowns (any column named "level" or "category") ──
        for col in ("level", "category"):
            if col in df.columns and df[col].notna().any():
                breakdown = (
                    df.groupby(col)["is_correct"]
                    .agg(accuracy="mean", n="count")
                    .round(4)
                    .to_dict(orient="index")
                )
                metrics[f"{col}_breakdown"] = breakdown

        return metrics

    def print_summary(self, metrics: dict) -> None:
        sep = "─" * 58
        print(f"\n{sep}")
        print(f"  Model    : {metrics.get('model', '?')}")
        print(f"  Dataset  : {metrics.get('dataset', '?')}")
        print(f"  Accuracy : {metrics.get('correct')}/{metrics.get('total')} "
              f"= {metrics.get('accuracy_pct')}")
        print(f"  Tokens   : {metrics.get('total_tokens', 0):,}  "
              f"(avg in={metrics.get('avg_input_tokens')} "
              f"out={metrics.get('avg_output_tokens')})")
        print(f"  Cost     : ${metrics.get('total_cost_usd', 0):.4f}")
        print(f"  Latency  : avg={metrics.get('avg_latency_s')}s  "
              f"p50={metrics.get('p50_latency_s')}s  "
              f"p95={metrics.get('p95_latency_s')}s")
        print(f"  Errors   : {metrics.get('errors')} "
              f"({metrics.get('error_rate', 0):.1%})")

        for col in ("level", "category"):
            key = f"{col}_breakdown"
            if key in metrics:
                print(f"  By {col.title()} :")
                for grp, v in metrics[key].items():
                    print(f"    {grp:<12} accuracy={v['accuracy']:.1%}  n={v['n']}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT STORE
# ─────────────────────────────────────────────────────────────────────────────

class ResultStore:
    """
    Persists results to disk. Structure:

      {base_dir}/
        {dataset}/
          {model}/
            responses.parquet   ← full row-level results
            metrics.json        ← aggregated metrics

    The exists() check is what prevents re-running the LLM.
    """

    def __init__(self, base_dir: str = "../results") -> None:
        self.base_dir = Path(base_dir)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _run_dir(self, dataset: str, model: str) -> Path:
        safe_ds    = dataset.lower().replace(" ", "_").replace("-", "_")
        safe_model = model.replace("/", "-")
        p = self.base_dir / safe_ds / safe_model
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Public API ────────────────────────────────────────────────────────

    def exists(self, dataset: str, model: str) -> bool:
        d = self._run_dir(dataset, model)
        return (d / "responses.parquet").exists() and (d / "metrics.json").exists()

    def save(self, dataset: str, model: str,
             results: list[Result], metrics: dict) -> Path:
        d = self._run_dir(dataset, model)
        pd.DataFrame([asdict(r) for r in results]).to_parquet(
            d / "responses.parquet", index=False
        )
        with open(d / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"  💾 Saved  → {d}/")
        return d

    def load_responses(self, dataset: str, model: str) -> pd.DataFrame:
        return pd.read_parquet(self._run_dir(dataset, model) / "responses.parquet")

    def load_metrics(self, dataset: str, model: str) -> dict:
        with open(self._run_dir(dataset, model) / "metrics.json") as fh:
            return json.load(fh)

    def load(self, dataset: str, model: str) -> tuple[pd.DataFrame, dict]:
        return self.load_responses(dataset, model), self.load_metrics(dataset, model)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER  (single model × single dataset)
# ─────────────────────────────────────────────────────────────────────────────

class Runner:
    """
    Executes one (model × dataset) run.
    Separated from the pipeline so it can be used standalone or in tests.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def run(self, queries: list[Query]) -> list[Result]:
        model   = self.client.config.name
        dataset = queries[0].dataset if queries else "?"
        total   = len(queries)
        results: list[Result] = []

        print(f"\n{'═' * 62}")
        print(f"  ▶  {model}  ×  {dataset}  ({total} queries)")
        print(f"{'═' * 62}")

        for i, q in enumerate(queries, 1):
            result = self.client.generate(q)
            results.append(result)

            status = "✓" if result.is_correct else "✗"
            err    = f"  ⚠ {result.error}" if result.error else ""
            print(
                f"  [{i:>4}/{total}] {status}  id={q.id:<20}  "
                f"tok={result.total_tokens:<6}  "
                f"cost=${result.cost_usd:.5f}  "
                f"t={result.time_seconds}s"
                f"{err}"
            )

        return results


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION PIPELINE  (main orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationPipeline:
    """
    Orchestrates N models × N datasets.

    Usage
    -----
    pipeline = EvaluationPipeline(
        models   = [ModelConfig("gpt-4o-mini"), ModelConfig("llama-3.1-70b")],
        datasets = {
            "gaia":      "datasets/processed/processed_gaia.parquet",
            "math_hard": "datasets/processed/processed_math_hard.parquet",
        },
        results_dir   = "results",
        system_prompt = "Answer concisely.",
    )
    pipeline.run()               # skips already-done runs automatically
    pipeline.run(force=True)     # re-run everything
    """

    def __init__(
        self,
        models: list[ModelConfig],
        datasets: dict[str, str],           # {dataset_name: path}
        results_dir: str = "../datasets/baseline",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.models        = models
        self.datasets      = datasets
        self.results_dir   = results_dir
        self.system_prompt = system_prompt

        self.loader    = DataLoader()
        self.evaluator = Evaluator()
        self.store     = ResultStore(results_dir)

    # ── Public API ────────────────────────────────────────────────────────

    def register_loader(self, name: str, fn: Callable[[str], list[Query]]) -> None:
        """Add a custom dataset loader at runtime."""
        self.loader.register(name, fn)

    def run(self, force: bool = False) -> list[dict]:
        """
        Run (or skip) every (model × dataset) combination.

        Parameters
        ----------
        force : bool
            If True, re-run even if results already exist on disk.

        Returns
        -------
        list[dict]
            All metrics dicts (from cache or freshly computed).
        """
        all_metrics: list[dict] = []

        # Pre-load datasets once (shared across all models)
        loaded_datasets: dict[str, list[Query]] = {}
        for ds_name, ds_path in self.datasets.items():
            loaded_datasets[ds_name] = self.loader.load(ds_name, ds_path)

        for model_cfg in self.models:
            for ds_name, queries in loaded_datasets.items():

                # ── Guard: skip if already done ──────────────────────────
                if not force and self.store.exists(ds_name, model_cfg.name):
                    print(
                        f"\n  ⏭  SKIP  {model_cfg.name} × {ds_name}  "
                        f"(results exist — pass force=True to override)"
                    )
                    _, metrics = self.store.load(ds_name, model_cfg.name)
                    self.evaluator.print_summary(metrics)
                    all_metrics.append(metrics)
                    continue

                # ── Run LLM ──────────────────────────────────────────────
                client  = LLMClient(model_cfg, system_prompt=self.system_prompt)
                runner  = Runner(client)
                results = runner.run(queries)

                # ── Evaluate ─────────────────────────────────────────────
                metrics = self.evaluator.compute(results)
                self.evaluator.print_summary(metrics)

                # ── Save (responses + metrics) ────────────────────────────
                self.store.save(ds_name, model_cfg.name, results, metrics)
                all_metrics.append(metrics)

        self._print_global_summary(all_metrics)
        return all_metrics

    # ── Private ───────────────────────────────────────────────────────────

    def _print_global_summary(self, all_metrics: list[dict]) -> None:
        if not all_metrics:
            return
        col_w = 22
        print(f"\n{'═' * 72}")
        print("  GLOBAL SUMMARY")
        print(f"{'═' * 72}")
        header = (
            f"  {'Dataset':<18} {'Model':<{col_w}} "
            f"{'Accuracy':>10} {'Tokens':>10} {'Cost':>10}"
        )
        print(header)
        print(f"  {'─' * 16} {'─' * col_w} {'─' * 10} {'─' * 10} {'─' * 10}")
        for m in all_metrics:
            print(
                f"  {m.get('dataset','?'):<18} "
                f"{m.get('model','?'):<{col_w}} "
                f"{m.get('accuracy_pct','?'):>10} "
                f"{m.get('total_tokens', 0):>10,} "
                f"${m.get('total_cost_usd', 0):>9.4f}"
            )
        total_cost = sum(m.get("total_cost_usd", 0) for m in all_metrics)
        total_tok  = sum(m.get("total_tokens", 0)   for m in all_metrics)
        print(f"  {'─' * 16} {'─' * col_w} {'─' * 10} {'─' * 10} {'─' * 10}")
        print(
            f"  {'TOTAL':<18} {'':<{col_w}} "
            f"{'':>10} {total_tok:>10,} ${total_cost:>9.4f}"
        )
        print(f"{'═' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Define models (add/remove freely) ─────────────────────────────
    MODELS = [
        ModelConfig(name="gpt-4o-mini",      temperature=0.1),
        # ModelConfig(name="llama-3.1-70b",  base_url="http://localhost:11434/v1"),
        # ModelConfig(name="gemini-1.5-flash", base_url="...", api_key="..."),
    ]

    # ── 2. Define datasets (add/remove freely) ────────────────────────────
    DATASETS = {
        "gaia":       "../datasets/processed/processed_gaia.parquet",
        # "swe_bench":  "../datasets/processed/processed_swe_bench.parquet",
        # "math_hard":  "../datasets/processed/processed_math_hard.parquet",
        # "agentbench": "../datasets/processed/processed_agentbench.parquet",
    }

    # ── 3. Build pipeline ─────────────────────────────────────────────────
    pipeline = EvaluationPipeline(
        models        = MODELS,
        datasets      = DATASETS,
        results_dir   = "../datasets/baseline",  
        system_prompt = DEFAULT_SYSTEM_PROMPT,
    )


    pipeline.run(force=False)