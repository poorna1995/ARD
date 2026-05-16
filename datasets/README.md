# Datasets

Benchmark data for agent training and evaluation. All processed tables share a **canonical schema**: `id`, `query`, `answer` (plus dataset-specific metadata).

Config: `configs/datasets.yaml`  
Loaders: `src/data/` (`MathLoader`, `HotpotLoader`, `MuSiQueLoader`, `MMLUProLoader`, `GAIALoader`)  
CLI download: `python scripts/download.py --dataset <name>`

---

## Directory layout

```
datasets/
├── raw/{dataset}/{split}/raw.parquet          # HuggingFace snapshot
├── processed/{dataset}/{split}/data.parquet   # cleaned, canonical columns
├── processed/gaia.parquet                     # GAIA (single validation split)
├── train_samples/                             # stratified subsamples for training
└── eval_samples/                              # stratified subsamples for evaluation
```

---

## Source datasets

| Key        | HuggingFace repo                   | Splits on disk        | Used for training | Used for eval       |
| ---------- | ---------------------------------- | --------------------- | ----------------- | ------------------- |
| `math`     | `EleutherAI/hendrycks_math`        | `train`, `test`       | yes (500 sample)  | yes (200 sample)    |
| `hotpot`   | `hotpot_qa`                        | `train`, `validation` | yes (500)         | yes (200)           |
| `musique`  | `dgslibisey/MuSiQue`               | `train`, `validation` | yes (500)         | yes (200)           |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro`               | `test`, `validation`  | **no**            | yes (200)           |
| `gaia`     | `gaia-benchmark/GAIA` (`2023_all`) | validation only       | **no**            | yes (165, full val) |

**Agent / prompt registry names** (e.g. in `prompts/prompts.py`): `math_hard`, `mmlu_pro`, `gaia`, `swe_bench_verified` — map to loader keys above where applicable (`math_hard` → MATH loader).

---

## How data is loaded and processed

1. **Download** — `BaseLoader._download()` pulls from HuggingFace into `datasets/raw/`.
2. **Process** — `_process()` renames to canonical columns, filters bad rows, writes `datasets/processed/`.
3. **Sample** — `src/utils/train_samples.py` builds fixed-size train/eval parquets (seed **42**).

**MATH-specific:** `answer` is taken from the last `\boxed{...}` in `solution`, then normalized with MATH `strip_string` (`src/math_latex.py`, `normalize_answers: true` in config). ~71% of answers are plain integers; ~29% are LaTeX (fractions, matrices, etc.).

**MuSiQue-specific:** `n_hops`, `hop_type`, `hop_name` are parsed from question IDs (e.g. `3hop2_...`). Unanswerable rows are dropped by default.

**MMLU-Pro-specific:** Only dataset with an **`options`** column (multiple choice). `answer` is the letter (A–J).

**GAIA-specific:** Validation only (gold answers). Optional **`file_name`** for attachments (download separately if testing file tools).

---

## Processed data (full)

| Path                                         | Shape      | Columns                                                                                                                    |
| -------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| `processed/math/train/data.parquet`          | 6,787 × 8  | `id`, `query`, `answer`, `level`, `type`, `solution`, `split`, `solution_length`                                           |
| `processed/math/test/data.parquet`           | 4,581 × 8  | same                                                                                                                       |
| `processed/hotpot/train/data.parquet`        | 90,447 × 6 | `id`, `query`, `answer`, `type`, `level`, `split`                                                                          |
| `processed/hotpot/validation/data.parquet`   | 7,405 × 6  | same                                                                                                                       |
| `processed/musique/train/data.parquet`       | 19,938 × 8 | `id`, `query`, `answer`, `split`, `n_hops`, `hop_type`, `hop_name`, `answerable`                                           |
| `processed/musique/validation/data.parquet`  | 2,417 × 8  | same                                                                                                                       |
| `processed/mmlu_pro/test/data.parquet`       | 12,032 × 8 | `id`, `query`, `answer`, `answer_index`, `options`, `category`, `cot_length`, `split`                                      |
| `processed/mmlu_pro/validation/data.parquet` | 70 × 8     | same                                                                                                                       |
| `processed/gaia.parquet`                     | 165 × 10   | `id`, `query`, `answer`, `level`, `annotator_steps`, `annotator_tools`, `file_name`, `steps_num`, `tool_num`, `time_taken` |

### Column map (canonical names)

Use these names in code — do not assume HF column names (`question`, `problem`, etc.).

| Role        | MATH                | Hotpot                  | MuSiQue                | MMLU-Pro              | GAIA          |
| ----------- | ------------------- | ----------------------- | ---------------------- | --------------------- | ------------- |
| Question    | `query`             | `query`                 | `query`                | `query`               | `query`       |
| Gold answer | `answer`            | `answer`                | `answer`               | `answer` (letter)     | `answer`      |
| Difficulty  | `level` (Level 1–5) | `level` (easy/med/hard) | `n_hops` (2–4)         | —                     | `level` (1–3) |
| Extra       | `type`, `solution`  | `type`                  | `hop_name`, `hop_type` | `options`, `category` | `file_name`   |

---

## Training data (constructed)

Built from **MATH + Hotpot + MuSiQue only** (not MMLU-Pro or GAIA).  
Stratified subsample: **500 rows per dataset**, `seed=42`, then stacked.

| File                                 | Shape          | Purpose                                                    |
| ------------------------------------ | -------------- | ---------------------------------------------------------- |
| `train_samples/math.parquet`         | 500 × 9        | MATH train subsample                                       |
| `train_samples/hotpot.parquet`       | 500 × 8        | Hotpot train subsample                                     |
| `train_samples/musique.parquet`      | 500 × 10       | MuSiQue train subsample                                    |
| `train_samples/combined_raw.parquet` | **1,500 × 15** | All three stacked (primary training table)                 |
| `train_samples/combined.parquet`     | 1,500 × …      | Same stack with global `training_id` only (`train_0000` …) |

### Per-file columns (training samples)

**`math.parquet` (500 × 9)**  
`id`, `query`, `answer`, `level`, `type`, `solution`, `split`, `solution_length`, `training_id`

- `training_id`: `math_0` … `math_499`
- Stratify: `level` × `type` (7 subjects × 5 levels)

**`hotpot.parquet` (500 × 8)**  
`id`, `query`, `answer`, `type`, `level`, `split`, `dataset_source`, `training_id`

- `dataset_source`: `hotpotqa`
- `training_id`: `hotpot_0` … `hotpot_499`
- Stratify: `level` × `type` (bridge / comparison)

**`musique.parquet` (500 × 10)**  
`id`, `query`, `answer`, `split`, `n_hops`, `hop_type`, `hop_name`, `answerable`, `dataset_source`, `training_id`

- `dataset_source`: `musique`
- `training_id`: `musique_0` … `musique_499`
- Stratify: 6 hop strata (2hop_linear, 3hop_linear, …, 4hop_branching); ~66% unique answers

**`combined_raw.parquet` (1,500 × 15)**  
Union of all columns from the three sources:

| Column                                         | MATH rows                            | Hotpot rows | MuSiQue rows |
| ---------------------------------------------- | ------------------------------------ | ----------- | ------------ |
| `id`, `query`, `answer`                        | filled                               | filled      | filled       |
| `level`, `type`, `solution`, `solution_length` | filled                               | NaN         | NaN          |
| `n_hops`, `hop_type`, `hop_name`, `answerable` | NaN                                  | NaN         | filled       |
| `dataset_source`                               | `math`                               | `hotpotqa`  | `musique`    |
| `dataset_training_id`                          | `math_*`                             | `hotpot_*`  | `musique_*`  |
| `training_id`                                  | `train_0000` … `train_1499` (global) | same        | same         |

NaNs in `combined_raw` are expected: dataset-specific fields are only populated for their source rows.

---

## Evaluation data (constructed)

**200 rows** per benchmark (stratified, `seed=42`), except GAIA (**full validation = 165**).

| File                           | Shape    | Source split         | Stratify by              |
| ------------------------------ | -------- | -------------------- | ------------------------ |
| `eval_samples/math.parquet`    | 200 × 8  | MATH `test`          | `level` × `type`         |
| `eval_samples/hotpot.parquet`  | 200 × 6  | Hotpot `validation`  | `level` × `type`         |
| `eval_samples/musique.parquet` | 200 × 8  | MuSiQue `validation` | hop strata               |
| `eval_samples/mmlu.parquet`    | 200 × 8  | MMLU-Pro `test`      | `category` (14 subjects) |
| `eval_samples/gaia.parquet`    | 165 × 10 | GAIA `validation`    | `level` (or full set)    |

Eval parquets use the same column names as processed data (no `training_id` except where noted in per-dataset train files).

---

## Quick access (Python)

```python
import pandas as pd

# Training (routing / multi-dataset experiments)
train = pd.read_parquet("datasets/train_samples/combined_raw.parquet")

# Per-benchmark eval
math_eval = pd.read_parquet("datasets/eval_samples/math.parquet")

# Full processed pool
math_train = pd.read_parquet("datasets/processed/math/train/data.parquet")
```

**Agent runs:** pass `expected_answer=row["answer"]` and set `dataset` to the prompt key (e.g. `math_hard` for MATH). For MATH, scoring uses LaTeX normalization via `evaluator.eval.is_correct(..., dataset="math_hard")`.

---

## Scripts (reference)

| Script                                 | Output                                                  |
| -------------------------------------- | ------------------------------------------------------- |
| `scripts/download.py --dataset <name>` | `raw/` + `processed/`                                   |
| `scripts/create_train_samples.py`      | `train_samples/{math,hotpot,musique,combined}.parquet`  |
| `scripts/create_eval_samples.py`       | `eval_samples/*.parquet`                                |
| `scripts/normalize_math_data.py`       | Re-apply MATH answer normalization on existing parquets |

<!-- --- -->

<!-- ## Notes

- **Required fields:** `id`, `query`, `answer` are non-null in all train/eval samples used for agents.
- **MMLU `options`:** stored as list-like values; format into the prompt when running multiple-choice agents.
- **GAIA attachments:** `file_name` is set for 38/165 eval rows; files live under `datasets/gaia_attachments/` after a separate download step. -->
