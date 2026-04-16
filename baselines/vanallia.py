import time
import json
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from openai import OpenAI  # works for GPT; swap client for QWEN/Llama

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
LLM_CONFIG = {
    "model": "gpt-4o-mini",          # swap: "qwen-2-72b", "gemini-1.5-flash", etc.
    "max_tokens": 1024,
    "temperature": 0.0,              # deterministic for eval
}

COST_PER_1K_TOKENS = {
    "gpt-4o-mini": {
        "input": 0.00015,   # $0.15 / 1M tokens
        "output": 0.00060
    },
    "gemini-1.5-flash": {
        "input": 0.00010,   # ~$0.10 / 1M tokens
        "output": 0.00030
    },
    "llama-3.1-70b": {
        "input": 0.00100,   # ~$1.00 / 1M tokens (varies by provider)
        "output": 0.00100
    },
    "qwen-2-72b": {
        "input": 0.00080,   # typical OpenRouter/Together range (~$0.8–1.2 / 1M)
        "output": 0.00080
    },
}
# ─────────────────────────────────────────────
# 2. DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class Query:
    id: str
    dataset: str          # GAIA | SWE-Bench | Math-Hard | AgentBench
    question: str
    ground_truth: str
    metadata: dict = field(default_factory=dict)

@dataclass
class Result:
    query_id: str
    dataset: str
    question: str
    ground_truth: str
    predicted: str
    is_correct: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    time_seconds: float
    model: str
    error: Optional[str] = None

# ─────────────────────────────────────────────
# 3. LOAD & UNIFY DATASETS
# ─────────────────────────────────────────────
def load_unified_dataset() -> list[Query]:
    """
    Replace each section with your actual dataset loader.
    Expected format: list of Query objects.
    """
    queries = []

    --- GAIA ---
    from datasets import load_dataset
    gaia = load_dataset("../datasets/processed/processed_gaia.parquet")
    for row in gaia["validation"]:
        queries.append(Query(
            id=row["task_id"], dataset="GAIA"`
            question=row["Question"], ground_truth=row["Final answer"]
        ))``

    # --- SWE-Bench Verified ---
    # for row in load_dataset("princeton-nlp/SWE-bench_Verified")["test"]:
    #     queries.append(Query(
    #         id=row["instance_id"], dataset="SWE-Bench",
    #         question=row["problem_statement"], ground_truth=row["patch"]
    #     ))

    # --- Math-Hard (Level 3-4) ---
    # for row in load_dataset("lighteval/MATH", "all")["test"]:
    #     if row["level"] in ["Level 3", "Level 4"]:
    #         queries.append(Query(
    #             id=row["problem"][:30], dataset="Math-Hard",
    #             question=row["problem"], ground_truth=row["solution"]
    #         ))

    # --- AgentBench ---
    # Load similarly from AgentBench splits

    # ── DUMMY DATA for smoke-test ──────────────────
    queries = [
        Query("g1", "GAIA",      "What is the capital of France?",       "Paris"),
        Query("m1", "Math-Hard",  "Solve: 2x + 5 = 13. What is x?",       "4"),
        Query("s1", "SWE-Bench",  "Fix the off-by-one error in this loop.","<patch>"),
        Query("a1", "AgentBench", "Book a flight from NYC to LA.",         "Completed"),
    ]
    return queries

# ─────────────────────────────────────────────
# 4. DIRECT LLM CALL  (vanilla baseline)
# ─────────────────────────────────────────────
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question as accurately and concisely as possible. "
    "Return only the final answer with no preamble."
)

def call_direct_llm(query: Query) -> Result:
    model = LLM_CONFIG["model"]
    start = time.perf_counter()
    error = None
    predicted = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=LLM_CONFIG["max_tokens"],
            temperature=LLM_CONFIG["temperature"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query.question},
            ],
        )
        predicted = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }
    except Exception as e:
        error = str(e)

    elapsed = time.perf_counter() - start

    # Cost calculation
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0, "output": 0})
    cost = (
        usage["prompt_tokens"]     / 1000 * rates["input"] +
        usage["completion_tokens"] / 1000 * rates["output"]
    )

    # Exact-match accuracy (swap with task-specific eval as needed)
    is_correct = predicted.strip().lower() == query.ground_truth.strip().lower()

    return Result(
        query_id=query.id,
        dataset=query.dataset,
        question=query.question,
        ground_truth=query.ground_truth,
        predicted=predicted,
        is_correct=is_correct,
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        cost_usd=round(cost, 6),
        time_seconds=round(elapsed, 3),
        model=model,
        error=error,
    )

# ─────────────────────────────────────────────
# 5. RUN EXPERIMENT
# ─────────────────────────────────────────────
def run_baseline(queries: list[Query]) -> list[Result]:
    results = []
    for i, q in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {q.dataset} | {q.id[:30]}")
        r = call_direct_llm(q)
        results.append(r)
        print(f"   ✓ correct={r.is_correct}  tokens={r.total_tokens}  "
              f"cost=${r.cost_usd:.5f}  time={r.time_seconds}s")
    return results

# ─────────────────────────────────────────────
# 6. EVALUATE & SAVE
# ─────────────────────────────────────────────
def evaluate(results: list[Result]):
    total   = len(results)
    correct = sum(r.is_correct for r in results)
    print("\n─── BASELINE SUMMARY ───────────────────────")
    print(f"  Model         : {results[0].model}")
    print(f"  Total queries : {total}")
    print(f"  Accuracy      : {correct}/{total} = {correct/total:.1%}")
    print(f"  Total tokens  : {sum(r.total_tokens for r in results):,}")
    print(f"  Total cost    : ${sum(r.cost_usd for r in results):.4f}")
    print(f"  Avg latency   : {sum(r.time_seconds for r in results)/total:.2f}s")

    # Per-dataset breakdown
    datasets = {r.dataset for r in results}
    for ds in sorted(datasets):
        ds_results = [r for r in results if r.dataset == ds]
        acc = sum(r.is_correct for r in ds_results) / len(ds_results)
        print(f"  [{ds}] accuracy={acc:.1%}  n={len(ds_results)}")

def save_results(results: list[Result], path="baseline_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    print(f"\n✅ Results saved to {path}")

    with open(path.replace(".csv", ".json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    queries = load_unified_dataset()
    print(f"Loaded {len(queries)} queries across {len({q.dataset for q in queries})} datasets\n")

    results = run_baseline(queries)
    evaluate(results)
    save_results(results)