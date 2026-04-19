# Baseline experiments

This folder holds the **unified baseline runner**: load a benchmark from processed parquet, run selected **modalities** on the **same query set**, score predictions, and write `responses.parquet` + `metrics.json` per `(dataset, modality, model)`.

## Prerequisites

- **Repo root** as working directory (paths below assume `research_work/`).
- **`OPENAI_API_KEY`** set in the environment (or `.env` loaded by the runner).
- Processed datasets under `datasets/processed/*.parquet` (or paths set in `run_baseline.py` → `DATASET_PATHS`).
- Optional: **`TAVILY_API_KEY`** for richer web search inside the ReAct tool stack.

## Environment setup

From the repository root:

```bash
cd /path/to/research_work
```

Using **uv** (recommended if you use `uv.lock`):

```bash
uv sync
```

Using a **venv**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

(Use whatever install path matches your `pyproject.toml`; the important part is that `openai`, `pandas`, `pyarrow`, etc. are available.)

Export credentials:

```bash
export OPENAI_API_KEY="sk-..."
```

Optional:

```bash
export TAVILY_API_KEY="tvly-..."
```

## CLI overview

```bash
python baselines/run_baseline.py --help
```

Main flags:

| Flag | Role |
|------|------|
| `--datasets` | One or more of: `gaia`, `mmlu_pro`, `math_hard`, `swe_bench_verified` |
| `--modalities` | e.g. `vanilla`, `zero_shot_cot`, `few_shot_cot`, `cot`, `react`, `multiagent`, `routellm`, `self_consistency_cot` |
| `--model` | e.g. `gpt-4o-mini`, `gpt-4o` |
| `--temperature` | Sampling temperature for single-completion modalities (see manifest notes for ReAct / self-consistency) |
| `--max-tokens` | Max completion tokens per **single** completion; ReAct per-step budget is `min(max_tokens, 512)` |
| `--sample-limit` | Cap rows loaded per dataset (after MMLU filters) |
| `--max-steps` | ReAct loop ceiling |
| `--tool-budget` | ReAct tool-call budget (passed into `react_operator`) |
| `--results-dir` | Output root (default `results/unified_baseline`) |

## Example commands

### Smoke: single dataset, few queries

**Vanilla**

```bash
python baselines/run_baseline.py --datasets gaia --modalities vanilla --model gpt-4o-mini --sample-limit 5
```

**Zero-shot CoT**

```bash
python baselines/run_baseline.py --datasets gaia --modalities zero_shot_cot --model gpt-4o-mini --sample-limit 5
```

**Few-shot CoT**

```bash
python baselines/run_baseline.py --datasets gaia --modalities few_shot_cot --model gpt-4o-mini --sample-limit 5
```

**ReAct**

```bash
python baselines/run_baseline.py --datasets gaia --modalities react --model gpt-4o-mini --sample-limit 5 --max-steps 6 --tool-budget 4
```

**Legacy `cot`** (still accepted on the CLI; **saved rows** use modality `zero_shot_cot` — see `run_manifest.json` and row `metadata`)

```bash
python baselines/run_baseline.py --datasets gaia --modalities cot --model gpt-4o-mini --sample-limit 5
```

### One run, comparable modalities (tune then freeze hyperparameters)

```bash
python baselines/run_baseline.py \
  --datasets gaia \
  --modalities zero_shot_cot few_shot_cot vanilla react \
  --model gpt-4o-mini \
  --temperature 0.1 \
  --max-tokens 1024 \
  --max-steps 6 \
  --tool-budget 4 \
  --sample-limit 20 \
  --results-dir results/unified_baseline
```

### RouteLLM baseline (optional)

```bash
python baselines/run_baseline.py \
  --datasets gaia \
  --modalities routellm \
  --model gpt-4o-mini \
  --sample-limit 5
```

### MMLU-Pro: list category values

```bash
python baselines/run_baseline.py --datasets mmlu_pro --list-mmlu-categories
```

## Outputs

After a successful run:

- **Per bucket:** `results/unified_baseline/<dataset>/<modality>/<model>/responses.parquet`
- **Per bucket:** `results/unified_baseline/<dataset>/<modality>/<model>/metrics.json`
- **Run manifest:** `{--results-dir}/run_manifest.json` — selection, dataset paths, counts, and short notes on modality normalization and generation policy

Module mode (equivalent entrypoint):

```bash
python -m baselines.run_baseline --datasets gaia --modalities vanilla --model gpt-4o-mini --sample-limit 5
```

## Key modules

| File | Purpose |
|------|---------|
| `run_baseline.py` | CLI, dataset discovery, `load_queries`, orchestration |
| `runner.py` | `run_selected`: modality dispatch, scoring, `UnifiedExperimentRecord` assembly |
| `prompt_templates.py` / `prompt_resolver.py` | Resolved prompts for non-ReAct modalities |
| `results.py` | Split by `(dataset, modality, model)`, write parquet + metrics |
| `../agents/react_operator.py` | ReAct tool loop |
| `../agents/cot_operator.py` | Explicit zero-shot / few-shot CoT operators (single completion) |

For repository-wide architecture, see `CODEBASE_ARCHITECTURE.md` at the repo root.
