




"""
Wei et al. 2022 — Chain-of-Thought Prompting Experiment
Optimised rewrite: parallel API calls, pre-compiled regex,
per-dataset result saving, bug fixes (path slicing + loop scope).
Logic is identical to the original.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
load_dotenv()


# ── optional sympy ────────────────────────────────────────────
try:
    from sympy import N, simplify, sympify
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("[WARN] sympy not installed — MATH-HARD will use string fallback only.")
    print("       Run: pip install sympy antlr4-python3-runtime")


# =============================================================
# 1. CONFIG
# =============================================================

LLM_CONFIG = {
    "model":       "gpt-4o-mini",
    "max_tokens":  2048,
    "temperature": 0.1,
}

COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini":      {"input": 0.00015, "output": 0.0006},
}

PATHS = {
    "gaia":      "../datasets/processed/processed_gaia.parquet",
    "mmlu_pro":  "../datasets/processed/processed_mmlu_pro.parquet",
    # "math_hard": "../datasets/processed/processed_math_hard.parquet",
}

SAMPLE_LIMIT: dict[str, Optional[int]] = {
    "gaia":      None,   # None = all rows
    "mmlu_pro":  200,
    # "math_hard": None,
}

OUTPUT_DIR = Path("results")
MAX_WORKERS = 8          # parallel API threads — tune to your rate limit
MAX_RETRIES = 3          # retries per failed API call
RETRY_BACKOFF = 2.0      # seconds between retries (doubles each attempt)

COT_TYPE = "few_shot"    # "few_shot" | "zero_shot"

client = OpenAI()        # reads OPENAI_API_KEY from env


# =============================================================
# 2. DATA CLASSES
# =============================================================

@dataclass
class Query:
    id:           str
    dataset:      str           # GAIA | MATH-HARD | MMLU-PRO
    question:     str
    ground_truth: str
    level:        Optional[str] = None
    options:      Optional[str] = None   # MMLU-PRO MCQ options
    subject:      Optional[str] = None
    metadata:     dict          = field(default_factory=dict)


@dataclass
class Result:
    query_id:       str
    dataset:        str
    question:       str
    ground_truth:   str
    full_reasoning: str
    predicted:      str
    is_correct:     bool
    input_tokens:   int
    output_tokens:  int
    total_tokens:   int
    cost_usd:       float
    time_seconds:   float
    model:          str
    cot_type:       str
    level:          Optional[str] = None
    subject:        Optional[str] = None
    error:          Optional[str] = None


# =============================================================
# 3. PRE-COMPILED REGEX  (compile once, reuse everywhere)
# =============================================================

_RE_THE_ANSWER   = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
_RE_BOXED        = re.compile(r"\\boxed\{([^}]+)\}")
_RE_EQ_END       = re.compile(r"=\s*([^\n=]+)\s*$", re.MULTILINE)
_RE_LETTER_MMLU  = re.compile(r"\b([A-Ja-j])\b")
_RE_LETTER_END   = re.compile(r"\b([A-E])\b\.?\s*$")
_RE_WHITESPACE   = re.compile(r"\s+")
_RE_IMPLICIT_MUL = re.compile(r"(\d)(pi\b|[a-df-wyzA-Z])")


# =============================================================
# 4. EXEMPLARS  (unchanged — paste your originals here)
# =============================================================

GAIA_EXEMPLARS = """\
Q: What is the capital of France?
A: France is a country in Western Europe. Its capital city is Paris. The answer is Paris.

Q: If a train travels 60 miles per hour for 2.5 hours, how far does it go?
A: Distance = speed × time = 60 × 2.5 = 150 miles. The answer is 150 miles.
"""

MMLU_EXEMPLARS = """\
Q: Which organelle is responsible for producing energy in a cell?
A. Nucleus  B. Ribosome  C. Mitochondria  D. Golgi apparatus
A: The mitochondria generate ATP through cellular respiration. The answer is C.

Q: What is the chemical formula for water?
A. CO2  B. H2O  C. NaCl  D. O2
A: Water consists of two hydrogen atoms and one oxygen atom. The answer is B.
"""

# MATH_EXEMPLARS = "..."  # add when re-enabling MATH-HARD

EXEMPLARS: dict[str, str] = {
    "GAIA":      GAIA_EXEMPLARS,
    "MMLU-PRO":  MMLU_EXEMPLARS,
    # "MATH-HARD": MATH_EXEMPLARS,
}


# =============================================================
# 5. ANSWER EXTRACTION
# =============================================================

def extract_answer(response_text: str, dataset: str) -> tuple[str, str]:
    """Return (full_reasoning, extracted_answer)."""
    full_reasoning = response_text.strip()

    match = _RE_THE_ANSWER.search(response_text)
    if match:
        raw = match.group(1).strip().rstrip(".")

        if dataset == "MMLU-PRO":
            lm = re.match(r"\(?([A-Ja-j])\)?[\.\):\s]?", raw)
            return full_reasoning, lm.group(1).upper() if lm else raw

        if dataset == "MATH-HARD":
            cleaned = (
                raw
                .replace("\u2212", "-").replace("\u00d7", "*")
                .replace("\u00f7", "/").replace("π", "pi")
                .replace("\u03c0", "pi").replace("²", "**2")
                .replace("³", "**3").replace("^", "**")
            )
            return full_reasoning, _RE_WHITESPACE.sub("", cleaned).strip()

        return full_reasoning, raw   # GAIA

    # ── fallbacks ────────────────────────────────────────────
    if dataset == "MMLU-PRO":
        letters = _RE_LETTER_MMLU.findall(response_text)
        if letters:
            return full_reasoning, letters[-1].upper()

    if dataset == "MATH-HARD":
        m = _RE_BOXED.search(response_text)
        if m:
            return full_reasoning, m.group(1).strip()
        m = _RE_EQ_END.search(response_text)
        if m:
            return full_reasoning, m.group(1).strip()

    lines  = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
    answer = lines[-1].rstrip(".") if lines else response_text.strip()
    return full_reasoning, answer


# =============================================================
# 6. NORMALISATION & ALIAS HELPERS
# =============================================================

GAIA_ALIASES: dict[str, list[str]] = {
    "united states":           ["us", "usa", "u.s.", "u.s.a.", "united states of america"],
    "united kingdom":          ["uk", "u.k.", "britain", "great britain"],
    "world war ii":            ["wwii", "ww2", "world war 2", "second world war"],
    "artificial intelligence": ["ai"],
    "united nations":          ["un", "u.n."],
}

def _build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in GAIA_ALIASES.items():
        norm_canon = _norm(canonical)
        for alias in aliases:
            alias_map[_norm(alias)] = norm_canon
    return alias_map

def _norm(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.rstrip(".,:;!")
    return _RE_WHITESPACE.sub(" ", text)

_ALIAS_MAP: dict[str, str] = _build_alias_map()

def _resolve_alias(text: str) -> str:
    n = _norm(text)
    return _ALIAS_MAP.get(n, n)

def _try_float(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


# =============================================================
# 7. DATASET-SPECIFIC EVALUATORS
# =============================================================

def eval_gaia(predicted: str, ground_truth: str) -> bool:
    p, g = _norm(predicted), _norm(ground_truth)
    if p == g:
        return True
    if _resolve_alias(p) == _resolve_alias(g):
        return True
    pn, gn = _try_float(predicted), _try_float(ground_truth)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    if g and (g in p or p in g):
        return True
    return False


def _preprocess_math(expr: str) -> str:
    result = (
        expr.replace("π", "pi").replace("\u03c0", "pi")
            .replace("\u2212", "-").replace("\u00d7", "*")
            .replace("\u00f7", "/").replace("²", "**2")
            .replace("³", "**3").replace("^", "**").strip()
    )
    result = _RE_IMPLICIT_MUL.sub(r"\1*\2", result)
    return _RE_WHITESPACE.sub("", result)

def _sympy_equal(e1: str, e2: str) -> Optional[bool]:
    if not SYMPY_AVAILABLE:
        return None
    a1, a2 = _preprocess_math(e1), _preprocess_math(e2)
    for fn in (sympify, parse_latex):
        try:
            diff = simplify(fn(a1) - fn(a2))
            if diff == 0 or abs(complex(N(diff))) < 1e-2:
                return True
        except Exception:
            continue
    return None

def _norm_math(text: str) -> str:
    text = text.lower().strip()
    return (
        _RE_WHITESPACE.sub("", text)
        .replace("^", "**").replace("×", "*").replace("÷", "/")
        .replace("π", "pi").rstrip(".")
    )

def eval_math(predicted: str, ground_truth: str) -> bool:
    r = _sympy_equal(predicted, ground_truth)
    if r is not None:
        return r
    if _norm_math(predicted) == _norm_math(ground_truth):
        return True
    pn, gn = _try_float(predicted), _try_float(ground_truth)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False

def eval_mmlu(predicted: str, ground_truth: str) -> bool:
    def _letter(text: str) -> str:
        text = text.strip()
        m = re.search(r"\b([A-Ja-j])\b", text)
        if m:
            return m.group(1).upper()
        return text[0].upper() if text and text[0].upper() in "ABCDEFGHIJ" else text.upper()
    return _letter(predicted) == _letter(ground_truth)

def compute_correctness(predicted: str, ground_truth: str, dataset: str) -> bool:
    if not predicted or not ground_truth:
        return False
    if dataset == "MMLU-PRO":
        return eval_mmlu(predicted, ground_truth)
    if dataset == "MATH-HARD":
        return eval_math(predicted, ground_truth)
    return eval_gaia(predicted, ground_truth)


# =============================================================
# 8. DATA LOADING
# =============================================================

def load_gaia(path: str, limit: Optional[int] = None) -> list[Query]:
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    print(f"[GAIA]      Loaded {len(df)} rows | Columns: {df.columns.tolist()}")
    return [
        Query(
            id           = str(row.get("id", i)),
            dataset      = "GAIA",
            question     = str(row["query"]),
            ground_truth = str(row["answer"]),
            level        = str(row.get("level", "")),
            metadata     = {"file": row.get("file_name"), "steps": row.get("steps_num")},
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]


def load_mmlu_pro(path: str, limit: Optional[int] = None) -> list[Query]:
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    print(f"[MMLU-PRO]  Loaded {len(df)} rows | Columns: {df.columns.tolist()}")
    queries: list[Query] = []
    for i, (_, row) in enumerate(df.iterrows()):
        options = row.get("options", None)
        if isinstance(options, list):
            labels  = [chr(65 + j) for j in range(len(options))]
            options_str = "  ".join(f"{lbl}) {opt}" for lbl, opt in zip(labels, options))
        else:
            options_str = options
        queries.append(Query(
            id           = str(row.get("id", i)),
            dataset      = "MMLU-PRO",
            question     = str(row.get("query", "")),
            ground_truth = str(row.get("answer", row.get("answer_index", ""))),
            level        = str(row.get("category", row.get("subject", ""))),
            options      = options_str,
            subject      = str(row.get("category", row.get("subject", ""))),
        ))
    return queries


def load_all() -> list[Query]:
    queries: list[Query] = []
    queries += load_gaia(PATHS["gaia"],     SAMPLE_LIMIT["gaia"])
    queries += load_mmlu_pro(PATHS["mmlu_pro"], SAMPLE_LIMIT["mmlu_pro"])
    # queries += load_math_hard(PATHS["math_hard"], SAMPLE_LIMIT.get("math_hard"))
    print(f"\nTotal: {len(queries)} queries\n")
    return queries


# =============================================================
# 9. PROMPT BUILDERS
# =============================================================

def build_few_shot_prompt(query: Query) -> list[dict]:
    exemplars = EXEMPLARS.get(query.dataset, GAIA_EXEMPLARS)
    system = (
        "You are an expert QA system using step-by-step logical reasoning.\n\n"
        "Rules:\n"
        "1. Understand the question fully.\n"
        "2. Solve using clear, logical steps, optimally.\n"
        "3. Apply correct formulas and facts.\n"
        "4. Ensure accuracy and consistency.\n"
        "5. Do not guess—reason carefully.\n\n"
        "Output (STRICT):\n"
        "- End EXACTLY with: The answer is <final_answer>.\n"
        "- No text after this line.\n"
        "- Final answer must be one of:\n"
        "  • single number\n"
        "  • option letter (A–J)\n"
        "  • one word\n"
        "  • short phrase\n"
        "- No explanation in the final answer line.\n"
    )
    question_text = query.question
    if (
        query.dataset == "MMLU-PRO"
        and query.options is not None
        and len(query.options) > 0
    ):
        options_str = "\n".join(
            f"{chr(65+i)}. {opt}"
            for i, opt in enumerate(query.options)
        )
        question_text = f"{query.question}\n{options_str}"

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"{exemplars}\n\nQ: {question_text}\nA:"},
    ]


def build_zero_shot_prompt(query: Query) -> list[dict]:
    question_text = query.question
    if query.dataset == "MMLU-PRO" and query.options:
        question_text = f"{query.question}\nOptions: {query.options}"
    return [
        {"role": "system", "content": "You are an expert reasoning system."},
        {"role": "user",   "content": f"Q: {question_text}\nA: Let's think step by step."},
    ]


# =============================================================
# 10. API CALL WITH RETRY
# =============================================================

def call_cot_llm(query: Query, cot_type: str = "few_shot") -> Result:
    model    = LLM_CONFIG["model"]
    messages = (
        build_few_shot_prompt(query)
        if cot_type == "few_shot"
        else build_zero_shot_prompt(query)
    )
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})

    full_reasoning = predicted = error = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    start = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model       = model,
                max_tokens  = LLM_CONFIG["max_tokens"],
                temperature = LLM_CONFIG["temperature"],
                messages    = messages,
            )
            raw_text = response.choices[0].message.content.strip()
            full_reasoning, predicted = extract_answer(raw_text, query.dataset)
            usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            }
            error = ""
            break
        except Exception as exc:
            error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    elapsed = time.perf_counter() - start
    cost = (
        usage["prompt_tokens"]     / 1000 * rates["input"] +
        usage["completion_tokens"] / 1000 * rates["output"]
    )

    return Result(
        query_id       = query.id,
        dataset        = query.dataset,
        question       = query.question,
        ground_truth   = query.ground_truth,
        full_reasoning = full_reasoning,
        predicted      = predicted,
        is_correct     = compute_correctness(predicted, query.ground_truth, query.dataset),
        input_tokens   = usage["prompt_tokens"],
        output_tokens  = usage["completion_tokens"],
        total_tokens   = usage["total_tokens"],
        cost_usd       = round(cost, 6),
        time_seconds   = round(elapsed, 3),
        model          = model,
        cot_type       = cot_type,
        level          = query.level,
        subject        = query.subject,
        error          = error or None,
    )


# =============================================================
# 11. PARALLEL EXPERIMENT RUNNER
# =============================================================

def run_cot_experiment(
    queries:    list[Query],
    cot_type:   str = "few_shot",
    max_workers: int = MAX_WORKERS,
) -> list[Result]:
    print(f"\n{'='*60}")
    print(f"  Wei et al. 2022 CoT — {cot_type.upper()}")
    print(f"  Model   : {LLM_CONFIG['model']}")
    print(f"  Queries : {len(queries)}")
    print(f"  Threads : {max_workers}")
    print(f"{'='*60}\n")

    results:  list[Optional[Result]] = [None] * len(queries)
    id_to_idx = {q.id: i for i, q in enumerate(queries)}

    def _run(q: Query) -> Result:
        return call_cot_llm(q, cot_type=cot_type)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_run, q): q for q in queries}
        for future in as_completed(future_map):
            q = future_map[future]
            r = future.result()
            results[id_to_idx[q.id]] = r
            completed += 1

            status = "✓" if r.is_correct else "✗"
            print(
                f"[{completed}/{len(queries)}] {r.dataset} | {r.query_id} | {status} "
                f"tokens={r.total_tokens}  cost=${r.cost_usd:.5f}  time={r.time_seconds}s"
            )
            if r.error:
                print(f"   ⚠ {r.error}")

    return [r for r in results if r is not None]


# =============================================================
# 12. PER-DATASET RESULT SAVING
# =============================================================



# =============================================================
# 13. EVALUATION REPORT
# =============================================================

def evaluate(results: list[Result]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])

    # Re-evaluate with strict per-dataset logic (non-destructive)
    df["is_correct_strict"] = df.apply(
        lambda row: compute_correctness(
            str(row["predicted"]), str(row["ground_truth"]), row["dataset"]
        ),
        axis=1,
    )

    model    = results[0].model
    cot_type = results[0].cot_type

    print(f"\n{'='*65}")
    print(f"  RESULTS — Wei et al. 2022 CoT  |  {cot_type.upper()}")
    print(f"  Model : {model}")
    print(f"{'='*65}")
    print(f"  Total queries : {len(df)}")
    print(f"  Accuracy (original / exact-match) : "
          f"{df['is_correct'].mean():.1%}  ({df['is_correct'].sum()}/{len(df)})")
    print(f"  Accuracy (strict  / per-dataset)  : "
          f"{df['is_correct_strict'].mean():.1%}  ({df['is_correct_strict'].sum()}/{len(df)})")
    print(f"  Total tokens  : {df['total_tokens'].sum():,}")
    print(f"  Total cost    : ${df['cost_usd'].sum():.4f}")
    print(f"  Avg latency   : {df['time_seconds'].mean():.2f}s")
    print(f"  Errors        : {df['error'].notna().sum()}")

    for dataset in sorted(df["dataset"].unique()):
        ddf = df[df["dataset"] == dataset].copy()
        print(f"\n{'─'*65}")
        print(f"  DATASET: {dataset}  (n={len(ddf)})")
        print(f"  Accuracy (exact-match) : {ddf['is_correct'].mean():.1%}")
        print(f"  Accuracy (strict)      : {ddf['is_correct_strict'].mean():.1%}")
        print(f"  Accuracy delta         : "
              f"{(ddf['is_correct_strict'].mean() - ddf['is_correct'].mean())*100:+.1f}pp")

        group_col = "subject" if dataset == "MMLU-PRO" else "level"
        if group_col in ddf.columns and ddf[group_col].notna().any():
            grp = (
                ddf.groupby(group_col, dropna=False)
                .agg(
                    correct   =("is_correct_strict", "sum"),
                    total     =("is_correct_strict", "count"),
                    acc       =("is_correct_strict", "mean"),
                    acc_exact =("is_correct", "mean"),
                    avg_tokens=("total_tokens", "mean"),
                    avg_cost  =("cost_usd", "mean"),
                )
                .sort_values("acc", ascending=False)
                .reset_index()
            )
            header = (
                f"  {group_col:<22} {'Correct':<9} {'Total':<7} "
                f"{'Acc(strict)':<13} {'Acc(exact)':<12} {'AvgTok'}"
            )
            print(f"\n{header}")
            print(f"  {'-'*21} {'-'*8} {'-'*6} {'-'*12} {'-'*11} {'-'*7}")
            for _, row in grp.iterrows():
                print(
                    f"  {str(row[group_col]):<22} "
                    f"{int(row['correct']):<9} "
                    f"{int(row['total']):<7} "
                    f"{row['acc']:<13.1%} "
                    f"{row['acc_exact']:<12.1%} "
                    f"{row['avg_tokens']:.0f}"
                )

        errors = ddf[ddf["error"].notna()]
        if not errors.empty:
            print(f"\n  ⚠  {len(errors)} API errors in {dataset}:")
            for _, row in errors.iterrows():
                print(f"     [{row['query_id']}] {row['error'][:80]}")

        failures = ddf[~ddf["is_correct_strict"]].head(3)
        if not failures.empty:
            print(f"\n  Sample failures ({dataset}):")
            for _, row in failures.iterrows():
                print(f"    Q  : {str(row['question'])[:80]}...")
                print(f"    GT : {row['ground_truth']}")
                print(f"    Pred: {row['predicted']}")
                print()
    _save_dataset(ddf, dataset)
    print(f"  Saved → {_BASE_OUT / dataset}")

    return df


BASE_OUT = Path("datasets/baseline/cot")
 
 
# ─── helpers ──────────────────────────────────────────────────────────────────
 
def _save_dataset(df: pd.DataFrame, dataset: str) -> None:
    """Persist per-dataset artefacts (non-blocking; prints on failure)."""
    out = _BASE_OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
 
    # results → CSV  (human-readable, small)
    result_cols = [
        "query_id", "question", "ground_truth", "predicted",
        "is_correct", "is_correct_strict", "dataset",
        "level", "subject", "total_tokens", "cost_usd",
        "time_seconds", "error",
    ]
    keep = [c for c in result_cols if c in df.columns]
    df[keep].to_csv(out / "results.csv", index=False)
 
    # full response → Parquet  (typed, compressed, fast to reload)
    df.to_parquet(out / "responses.parquet", index=False, compression="zstd")
 
 
def _acc(series: pd.Series) -> str:
    """'75.0%  (30/40)' from a bool series."""
    n, total = series.sum(), len(series)
    return f"{n / total:.1%}  ({int(n)}/{total})"
 
 
def _print_group_table(
    ddf: pd.DataFrame,
    group_col: str,
) -> None:
    grp = (
        ddf.groupby(group_col, dropna=False)
        .agg(
            correct   =("is_correct_strict", "sum"),
            total     =("is_correct_strict", "count"),
            acc       =("is_correct_strict", "mean"),
            acc_exact =("is_correct", "mean"),
            avg_tokens=("total_tokens", "mean"),
        )
        .sort_values("acc", ascending=False)
        .reset_index()
    )
    header = (
        f"  {group_col:<22} {'Correct':<9} {'Total':<7} "
        f"{'Acc(strict)':<13} {'Acc(exact)':<12} {'AvgTok'}"
    )
    print(f"\n{header}")
    print(f"  {'-'*21} {'-'*8} {'-'*6} {'-'*12} {'-'*11} {'-'*7}")
    for _, row in grp.iterrows():
        print(
            f"  {str(row[group_col]):<22} "
            f"{int(row['correct']):<9} "
            f"{int(row['total']):<7} "
            f"{row['acc']:<13.1%} "
            f"{row['acc_exact']:<12.1%} "
            f"{row['avg_tokens']:.0f}"
        )


# =============================================================
# 14. ENTRY POINT
# =============================================================

if __name__ == "__main__":
    all_queries = load_all()

    # Run experiments per dataset and collect all results
    all_results: list[Result] = []

    for dataset_name in ["GAIA", "MMLU-PRO"]:
        queries = [q for q in all_queries if q.dataset == dataset_name]
        if not queries:
            print(f"[SKIP] No queries found for {dataset_name}")
            continue

        print(f"\n{'#'*60}")
        print(f"  Running: {dataset_name}  ({len(queries)} queries)")
        print(f"{'#'*60}")

        results = run_cot_experiment(queries, cot_type=COT_TYPE, max_workers=MAX_WORKERS)
        all_results.extend(results)

    # Save per-dataset results
    print(f"\n{'='*60}")
    print("  Saving results …")
    saved_paths = save_results(all_results, COT_TYPE)

    # Print full evaluation report
    df = evaluate(all_results)