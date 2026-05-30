# Selective QCE — Phase 2 requirements & behavior

**Status:** Phase 2 (opt-in). **Default pipeline is unchanged** unless you pass `--selective-qce`.

---

## 1. Problem statement

Full QCE routing (`cvec5_emb` + `hgbm_cvec5_emb_soft_kl`) gives the best internal utility regret (~0.19 on test) but requires **LLM decompose on every query** (~0.22–0.26 mUSD/query).

Phase 2 adds an **optional** path:

1. Route with **cheap** features (embedding only + `hgbm_emb_only_soft_kl`).
2. If the cheap router is **confident**, use its agent and **skip decompose**.
3. If **uncertain**, run decompose + graph C(Q) and route with the **full** production router.

Goal: reduce decompose spend while keeping regret close to always-full QCE.

---

## 2. Scope

### In scope

| Item | Description |
|------|-------------|
| Config | Central paths, τ grid, decompose cost, output dirs (`routing/config.py`) |
| Core logic | Gate, hybrid routing, metrics (`routing/selective.py`) |
| Tuning CLI | Sweep τ on val, write CSV + recommended τ JSON (`scripts/tune_selective_qce.py`) |
| Opt-in orchestrator | `--selective-qce` (+ τ / router paths); **default off** |
| Internal splits | `train` / `val` / `test` with prebuilt parquets (simulation + route-only) |
| Eval sets (GAIA, …) | Embedding build for all; decompose **only** for uncertain ids when `--build-features` |

### Out of scope (later)

- Distilled `emb → dim_*` model (no LLM).
- Per-dataset τ tables (GAIA vs MATH).
- Including decompose cost in **training** utility (eval/tuning only for now).
- Changing `ROUTER_MODEL_PATH` or default `orchestrator` behavior without the flag.

---

## 3. Dependencies (must exist before Phase 2 runs)

| Artifact | Path | Role |
|----------|------|------|
| Cheap router | `models/router/graph_main/hgbm_emb_only_soft_kl.joblib` | Gate + confident routing |
| Full router | `models/router/graph_main/hgbm_cvec5_emb_soft_kl.joblib` | Uncertain routing |
| PCA | `models/query_embeddings/pca_16.joblib` | Embeddings |
| Train norm | `models/qce_graph/train_norm.json` | C(Q) normalization |
| Oracle (offline metrics) | `results/experiments/classifier_baselines_cvec5_emb/` or default oracle CSV | Regret / EM |
| Prebuilt features (splits) | `datasets/qce_features/complexity_record_{split}.parquet`, `query_embeddings_{split}.parquet` | Offline τ sweep |
| Decompose cache (eval) | `datasets/decomposer_cache/qce_{tag}_plans.jsonl` | Partial decompose on GAIA |

Train cheap router if missing:

```bash
uv run python -m routing.soft_train --classifier hgbm --feature-set emb_only --split both --seed 42 --save
```

---

## 4. Configuration (`routing/config.py`)

| Constant | Meaning |
|----------|---------|
| `SELECTIVE_CHEAP_ROUTER_PATH` | `hgbm_emb_only_soft_kl.joblib` |
| `SELECTIVE_FULL_ROUTER_PATH` | Same as `PRIMARY_ROUTER_PATH` |
| `SELECTIVE_DECOMPOSE_COST_USD` | Default 0.00025 (tune for Table 6) |
| `SELECTIVE_TAU_GRID` | τ values for sweep |
| `DEFAULT_SELECTIVE_TAU` | 0.50 (starting point from internal sweep) |
| `SELECTIVE_QCE_OUT_DIR` | `results/experiments/selective_qce/` |
| `SELECTIVE_TAU_JSON` | Saved recommendation from tune script |

User overrides: CLI `--selective-tau`, `--selective-cheap-router`, `--selective-full-router`.

---

## 5. Gate logic (confident vs uncertain)

**Inputs (from cheap router only):**

- `max_prob_emb` — max of `p_react`, `p_cot`, `p_raw`, `p_multiagent` after `emb_only` predict.
- `margin_top2_emb` — top1 − top2 probability.

**Config:**

- `tau` (required): e.g. 0.50.
- `tau_margin` (optional): if set, require **both** `max_prob_emb >= tau` **and** `margin_top2_emb >= tau_margin`.

**Rules:**

```text
CONFIDENT  iff  max_prob_emb >= tau  AND  (tau_margin is None OR margin_top2_emb >= tau_margin)
UNCERTAIN  otherwise
```

**On CONFIDENT:**

- `used_decompose = 0`
- `selective_path = "cheap"`
- Final agent = cheap router argmax (`router_pred_cheap` → `router_pred`)
- Final probabilities = cheap router `p_*`

**On UNCERTAIN:**

- `used_decompose = 1`
- `selective_path = "full"`
- Final agent = full router argmax
- Final probabilities = full router `p_*`

**Columns written on routed frame:**

| Column | Description |
|--------|-------------|
| `selective_path` | `"cheap"` \| `"full"` |
| `used_decompose` | 0 \| 1 |
| `max_prob_emb` | Cheap router confidence |
| `margin_top2_emb` | Cheap margin |
| `selective_tau` | τ used for this run |
| `router_pred` / `assigned_agent` | Final agent (orchestrator-compatible) |
| `p_*` | Final probabilities |
| `max_prob` | From **final** router row |
| `router_experiment` | `selective_qce` |

---

## 6. Functional modules

### 6.1 `routing/selective.py`

| Function | Behavior |
|----------|----------|
| `SelectiveGateConfig` | Frozen dataclass: τ, optional τ_margin, router paths, decompose_cost |
| `confident_mask(df, config)` | Boolean series |
| `attach_cheap_predictions(df, cheap_obj)` | Adds `*_cheap` cols + `max_prob_emb` |
| `attach_full_predictions(df, full_obj)` | Adds full-router preds (all rows; used only where gate says full) |
| `apply_selective_gate(df, config)` | Merges cheap/full into final `router_pred`, `p_*`, path cols |
| `route_selective_split(split, config)` | Load `cvec5_emb` split → gate → return routed df |
| `selective_metrics(df, oracle, config)` | Regret, EM, % skip decompose, mean total USD |
| `sweep_tau(split, taus, config)` | Table of metrics per τ |
| `selective_build_and_route(base, tag, config, …)` | GAIA/eval: embeddings all → gate → decompose subset → route |

### 6.2 `scripts/tune_selective_qce.py`

| Step | Action |
|------|--------|
| 1 | Sweep `SELECTIVE_TAU_GRID` on **val** |
| 2 | Print table: τ, % skip decompose, regret, EM, mean total USD |
| 3 | Pick recommended τ (among τ with val regret ≤ always-full + 0.01, choose closest to `DEFAULT_SELECTIVE_TAU`=0.50) |
| 4 | Write `SELECTIVE_TAU_JSON` and `tau_sweep_val.csv` |
| 5 | Optional `--eval-test`: one-row test metrics for recommended τ only |

### 6.3 `orchestrator/pipeline.py` (opt-in only)

New flags (all optional; **default = current behavior**):

| Flag | Default |
|------|---------|
| `--selective-qce` | off |
| `--selective-tau` | `DEFAULT_SELECTIVE_TAU` (0.50) |
| `--selective-tau-margin` | None |
| `--selective-cheap-router` | `SELECTIVE_CHEAP_ROUTER_PATH` |
| `--selective-full-router` | `SELECTIVE_FULL_ROUTER_PATH` |

When `--selective-qce` and **not** `--routes-path`:

- **QCE split** (`--split val|test`): `route_selective_split` (no API calls if parquets exist).
- **Eval parquet** (`--dataset gaia`): `selective_build_and_route` if `--build-features`; else require embeddings + complexity parquets (or build embeddings + partial decompose as needed).

When `--routes-path`: selective flag ignored (pre-routed file).

`--router-path` ignored under selective (uses cheap/full pair from config/flags).

---

## 7. Eval / GAIA build behavior (selective + `--build-features`)

```text
FOR each query in eval set:
  1. Encode query → emb_* (all queries)     # no LLM
  2. Cheap router → max_prob_emb

FOR queries with max_prob_emb < tau:
  3. decompose_batch (cache-aware) → dim_*  # LLM only for uncertain

FOR all queries:
  4. apply_selective_gate → final router_pred
```

Efficiency: decompose count = |uncertain|, not |dataset|.

---

## 8. Metrics (tuning & reporting)

Same oracle harness as `routing/baselines.py`:

- **Primary:** `mean_utility_regret` (λ=25, agent cost from oracle).
- **Secondary:** `exec_em`, `pct_skip_decompose`, `mean_total_usd` = agent + `used_decompose * SELECTIVE_DECOMPOSE_COST_USD`.

Baselines in sweep tables:

- `always_cheap` — 100% skip decompose
- `always_full` — 0% skip decompose

---

## 9. Acceptance criteria

1. Default `uv run python orchestrator/pipeline.py --dataset gaia --route-only` behavior unchanged (full router, full features path).
2. `uv run python scripts/tune_selective_qce.py` produces val sweep CSV + `recommended_tau.json`.
3. `uv run python orchestrator/pipeline.py --split test --route-only --selective-qce` routes 100 rows with `selective_path` / `used_decompose` columns.
4. Documented τ tuning: **val only**; test reported once.

---

## 10. Route scoring (lookup EM / mUSD)

After route-only, the pipeline scores from precomputed outcomes (no agent re-runs):

- **Live eval:** `routing/score_routes.py` joins `router_pred` to always-{agent} baselines under `LIVE_EVAL_BASELINE_ROOT`.
- **Internal splits:** same module joins to `oracle_results` (as soft_train).

```bash
uv run python orchestrator/pipeline.py --split test --route-only   # auto lookup score
uv run python -m routing score-routes --routes-path routes.parquet --dataset gaia
```

Disable with `--no-score-lookup`.

---

## 11. Commands (quick reference)

```bash
# Tune τ on val
uv run python scripts/tune_selective_qce.py

# Internal test route-only (selective)
uv run python orchestrator/pipeline.py --split test --route-only --selective-qce --selective-tau 0.50

# GAIA: build + selective route
uv run python orchestrator/pipeline.py --dataset gaia --build-features --route-only \
  --selective-qce --selective-tau 0.50 \
  --output_path results/experiments/selective_qce/gaia_routes.parquet
```

---

## 11. Design diagram

```mermaid
flowchart TB
  subgraph default [Default pipeline - unchanged]
    D1[build_eval_features ALL] --> D2[cvec5 router] --> D3[agent]
  end

  subgraph phase2 [Phase 2 - --selective-qce]
    P1[build emb ALL] --> P2[emb_only router]
    P2 --> G{confident?}
    G -->|yes| P3[cheap agent]
    G -->|no| P4[decompose + cvec5 router]
    P4 --> P5[full agent]
    P3 --> P6[agent]
    P5 --> P6
  end
```
