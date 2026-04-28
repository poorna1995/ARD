# Baselines README

This directory contains the unified baseline runner used to evaluate multiple
prompting/routing methods on shared benchmark datasets.

Supported datasets:

- `gaia`
- `mmlu_pro`
- `math_hard`
- `swe_bench_verified`

Supported baseline methods (modalities):

- `vanilla`
- `zero_shot_cot`
- `few_shot_cot`
- `react`
- `multiagent`
- `routellm`

---

## 1) Installation

Run from repository root (`research_work/`).

### Option A: uv (recommended)

```bash
uv sync
```

### Option B: venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Required environment variables

```bash
export OPENAI_API_KEY="sk-..."
```

Optional (recommended for richer ReAct tooling):

```bash
export TAVILY_API_KEY="tvly-..."
```

If you run Llama models:

```bash
export GROQ_API_KEY="gsk_..."
```

---

## 2) Quickstart (Most Important)

This section is the fastest way to run and verify baselines end-to-end.

### Step 1: Check CLI

```bash
python baselines/run_baseline.py --help
```

### Step 2: Run a tiny smoke test (Vanilla on GAIA)

```bash
python baselines/run_baseline.py \
  --datasets gaia \
  --modalities vanilla \
  --model gpt-4o-mini \
  --sample-limit 5
```

### Step 3: Compare multiple methods on the same queries

```bash
python baselines/run_baseline.py \
  --datasets gaia \
  --modalities vanilla zero_shot_cot few_shot_cot react \
  --model gpt-4o-mini \
  --sample-limit 20 \
  --results-dir results/unified_baseline
```

### Step 4: Run RouteLLM baseline (optional)

```bash
python baselines/run_baseline.py \
  --datasets gaia \
  --modalities routellm \
  --model gpt-4o-mini \
  --sample-limit 20
```

### Step 5: Where outputs appear

Each run writes:

- `results/unified_baseline/<dataset>/<modality>/<model>/responses.parquet`
- `results/unified_baseline/<dataset>/<modality>/<model>/metrics.json`
- `results/unified_baseline/run_manifest.json`

---

## 3) Running All Experiments

Use this pattern to run the same model across all supported datasets and
baseline modalities:

```bash
python baselines/run_baseline.py \
  --datasets gaia mmlu_pro math_hard swe_bench_verified \
  --modalities vanilla zero_shot_cot few_shot_cot react multiagent routellm \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --results-dir results/unified_baseline
```

Helpful notes:

- Start with `--sample-limit` during debugging to control cost and latency.
- For `mmlu_pro`, you can inspect available categories:
  ```bash
  python baselines/run_baseline.py --datasets mmlu_pro --list-mmlu-categories
  ```
- Equivalent module entrypoint:
  ```bash
  python -m baselines.run_baseline --datasets gaia --modalities vanilla --model gpt-4o-mini
  ```

---

## 4) Results

The table below is populated from `results1/unified_baseline/**/metrics.json`.

### GAIA

| Modality      | Model                   |   Level1 |   Level2 |   Level3 | Accuracy | Avg Tokens / Query | Avg Latency (s) | Cost (USD) |
| ------------- | ----------------------- | -------: | -------: | -------: | -------: | -----------------: | --------------: | ---------: |
| vanilla       | gpt-4o-mini             | 0.056604 | 0.046512 | 0.000000 | 0.042424 |            241.691 |           0.610 |   0.006257 |
| zero_shot_cot | gpt-4o-mini             | 0.132075 | 0.093023 | 0.000000 | 0.090909 |            295.527 |           1.573 |   0.012926 |
| few_shot_cot  | gpt-4o-mini             | 0.113208 | 0.093023 | 0.000000 | 0.084848 |            349.424 |           1.640 |   0.013951 |
| routellm_mf   | gpt-4o-mini + gpt-4o    | 0.056604 | 0.104651 | 0.000000 | 0.072727 |            216.648 |           1.873 |   0.050597 |
| vanilla       | llama-3.3-70b-versatile | 0.094340 | 0.034884 | 0.038462 | 0.054545 |            272.782 |           1.982 |   0.000000 |
| zero_shot_cot | llama-3.3-70b-versatile | 0.075472 | 0.011628 | 0.038462 | 0.036364 |            205.533 |           1.089 |   0.000000 |

### MMLU_PRO

| Modality      | Model       | Accuracy | Avg Tokens / Query | Avg Latency (s) | Cost (USD) | Dataset |
| ------------- | ----------- | -------: | -----------------: | --------------: | ---------: | ------: |
| vanilla       | gpt-4o-mini | 0.488889 |            267.526 |           0.633 |   0.010952 |
| zero_shot_cot | gpt-4o-mini | 0.596296 |            443.656 |           2.858 |   0.032435 |
| few_shot_cot  | gpt-4o-mini | 0.592593 |            520.237 |           2.243 |   0.033780 |

### MATH_HARD

| Modality     | Model                | Accuracy | Avg Tokens / Query | Avg Latency (s) | Cost (USD) | Dataset |
| ------------ | -------------------- | -------: | -----------------: | --------------: | ---------: | ------: |
| vanilla      | gpt-4o-mini          | 0.130000 |            192.947 |           0.976 |   0.011675 |
| few_shot_cot | gpt-4o-mini          | 0.710000 |            750.587 |           8.671 |   0.096109 |
| routellm     | routellm             | 0.246667 |            295.720 |           2.511 |   0.502936 |
| routellm_mf  | gpt-4o-mini + gpt-4o | 0.400000 |            174.200 |           3.121 |   0.004525 |

Notes:

- GAIA level metrics come from `metadata.gaia_level_metrics` in each metrics file.
- `math_hard/routellm` appears in two result buckets (`routellm` and `routellm_mf`) with different run sizes/configs, so both are listed.
- If you only want one official number per modality/model, pick a single canonical run path and keep that row only.

---

## 5) Project Structure

Proper folder structure for the baseline module:

```text
baselines/
├── __init__.py
├── README.md
├── constants.py          # Allowed datasets/modalities/models + validators
├── prompt_matrix.py      # Prompt matrix/configuration helpers
├── prompt_resolver.py    # Resolves prompts per dataset + modality
├── prompt_templates.py   # Prompt templates used by baseline methods
├── results.py            # Writes responses.parquet + metrics.json
├── run_baseline.py       # Main CLI entrypoint to run experiments
├── run_manifest.py       # Run manifest creation/serialization utilities
├── runner.py             # Core run loop, dispatch, extraction, scoring
├── schema.py             # Shared data schema/record definitions
└── routellm/
    ├── __init__.py
    ├── README.me
    └── routellm.py       # RouteLLM integration helpers
```

Related (outside this folder):

- `agents/react_operator.py`: ReAct execution and tool loop integration.

---

For broader repository architecture, see `CODEBASE_ARCHITECTURE.md` at the repo root.
