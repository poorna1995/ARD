# Module Catalog — Input · Midprocess · Output

Every **active** module (excluding `old/`, notebooks, gitignored-only scripts).  
Zone: **IN** = input · **MID** = midprocess · **OUT** = output.

**Critical:** this project has **two evaluation universes** with **different baseline systems** (see §1). Refactors must never mix them.

---

## 1. Two evaluation universes + baseline types

### 1.1 Universe A — Internal QCE (train / val / test)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Train router, pick hyperparams, paper Figures 1–4, internal regret |
| **Queries** | 808-row pool: `qce_train.csv` (train), `qce_val.csv` (val), `qce_internal_test.csv` (test) |
| **Features** | Prebuilt parquets: `datasets/qce_features/complexity_record_{split}.parquet`, `query_embeddings_{split}.parquet` |
| **Labels** | `oracle_agent`, `p_*` from utility argmax on internal oracle |
| **Outcome source** | **`datasets/train_samples/v1/oracle_results1.csv`** (long: one row per `training_id` × agent) |
| **How always-agent works** | **Virtual** — no separate runs; `oracle_outcome_matrix()` pivots CSV to wide `correct_{agent}`, `cost_{agent}` |
| **Score entrypoint** | `score_routed(..., split="test")` → `outcome_source: "oracle"` |
| **CLI** | `routing analyze --split test`, `routing train`, `routing soft-train`, `orchestrator --split test --route-only` |

### 1.2 Universe B — Live eval (eval_samples)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Table 5, live EM/mUSD, oracle bounds, deployment validation |
| **Queries** | `datasets/eval_samples/{gaia,hotpot,musique,math,mmlu}.parquet` (~165–200 each; GAIA ~165) |
| **Features** | Per-tag parquets under `datasets/qce_features/{tag}/` or built with `--build-features` |
| **Labels** | No training labels; gold = `expected_answer` in eval parquet |
| **Outcome source** | **Real agent runs** — four always-{agent} orchestrator jobs per dataset |
| **On-disk layout** | `results/orchestrator/graph_main_tuned/{folder}/{name}_baseline_{agent}/pipeline_results.parquet` |
| **How always-agent works** | **Physical** — run `orchestrator` once per agent on full eval set |
| **Score entrypoint** | `score_routed(..., dataset="gaia")` → `outcome_source: "live_baselines"` |
| **CLI** | `routing benchmark route-dist`, `routing score-routes --dataset gaia`, `routing benchmark oracle-bounds` |

### 1.3 Baseline terminology (four different meanings)

| Name | Universe | What it is | Module | Output |
|------|----------|------------|--------|--------|
| **Oracle outcomes (internal)** | A | Long CSV of 4-agent results on QCE pool | `analysis.oracle_outcome_matrix` | Wide matrix per `training_id` |
| **Always-{agent} (internal virtual)** | A | Metrics if you always pick one agent | `analysis.baseline_strategies` | `StrategyMetrics` list |
| **Always-{agent} (live physical)** | B | Full eval run with fixed agent | `orchestrator` + `oracle_bounds.load_baseline_agent` | `pipeline_results.parquet` × 4 |
| **Classifier baselines (Step 2)** | A | logreg/LGBM vs frozen HGBM on `cvec5_emb` | `routing/baselines.py` | `baselines_all.csv` |
| **Training baselines B0** | A | Majority / per-dataset mode label predictors | `router.baseline_always_majority` | `EvalResult` in train CSV |
| **Pre-routed baseline routes** | B | Parquet with every row assigned to one agent (no router) | manual / `baseline_routes/` | `*_baseline_{agent}_routes.parquet` |
| **Oracle bounds (live)** | B | Upper/lower EM from live always-agent runs | `oracle_bounds.py` | `oracle_bounds_long.csv` → Table 5 |

### 1.4 Scoring dispatch (must preserve)

```
score_routed(routes, split=...)     → outcomes_from_oracle()     → oracle_results1.csv
score_routed(routes, dataset=...) → outcomes_from_baselines()  → graph_main_tuned/*_baseline_*/
```

Orchestrator route-only: `score_lookup=True` uses same dispatch via `score_dataset` / `score_split` derived from CLI.

---

## 2. Config & entrypoints

### `config/paths.py`
| | |
|---|---|
| **IN** | `RESEARCH_USE_NEW_PATHS` env flag |
| **MID** | Canonical repo paths; legacy vs new layout switch |
| **OUT** | `REPO_ROOT`, `complexity_record_path`, `embeddings_parquet_path`, … |
| **Universe** | Both |

### `config/settings.py`
| | |
|---|---|
| **IN** | Env vars, `config/paths.py` |
| **MID** | Resolved model paths, feature flags |
| **OUT** | `router_model_path()`, deployment helpers |
| **Universe** | Both |

### `config/logging.py`, `config/secrets.py`, `config/llm_retry.py`
| | |
|---|---|
| **IN** | Env / `.env` |
| **MID** | Structured logging, secret lookup, LLM retry wrapper |
| **OUT** | Shared ops utilities (Phase 6) |
| **Universe** | Both (orchestrator + QCE decompose) |

### `routing/config.py`
| | |
|---|---|
| **IN** | None (constants only) |
| **MID** | Path resolution, feature-set specs, router role aliases, selective QCE defaults |
| **OUT** | Imported constants: `ROUTER_MODEL_PATH`, `EVAL_SAMPLES_DIR`, `LIVE_EVAL_BASELINE_ROOT`, `SPLIT_CSV`, … |
| **Deps** | stdlib only |
| **Universe** | Both — defines paths for A and B |

### `routing/__init__.py`
| | |
|---|---|
| **IN** | None |
| **MID** | Public API re-exports |
| **OUT** | Stable symbols for orchestrator / external scripts |
| **Deps** | `routing.config`, `routing.router`, `routing.analysis` |

### `routing/__main__.py`
| | |
|---|---|
| **IN** | `sys.argv` |
| **MID** | CLI dispatch to subcommands |
| **OUT** | Side effects (train, analyze, benchmark, …) |
| **Subcommands** | default→`router`, `soft-train`, `baselines`, `analyze`, `benchmark`, `score-routes`, `figures` |

---

### `routing/cli.py`
| | |
|---|---|
| **IN** | `sys.argv` |
| **MID** | Unified CLI; default subcommand = `verify` |
| **OUT** | Delegates to `routing.train.experiments`, `routing.verify`, … |
| **Entry** | `research-route` console script |

### `routing/verify.py`
| | |
|---|---|
| **IN** | Production router joblib, golden test fixtures |
| **MID** | Smoke checks on imports, paths, inference |
| **OUT** | Exit 0/1 for CI gate |

### `api/main.py` (optional)
| | |
|---|---|
| **IN** | HTTP request with query text |
| **MID** | FastAPI wrapper around `RuntimeRouter` |
| **OUT** | JSON route prediction |
| **Install** | `uv sync --group api` |

---

## 3. Routing — data, features, train, infer

Phase 3 split the former ~1800-line `routing/router.py` god module.  
**`routing/router.py`** is now a **~164-line compatibility shim** re-exporting the public API.  
Archived monolith: `archive/refactor_2026/routing/router.py`.

### `routing/data/splits.py`
| | |
|---|---|
| **IN** | QCE label CSVs (`SPLIT_CSV`), complexity parquets, optional emb/heur parquets |
| **MID** | Merge labels + features; column alias normalization |
| **OUT** | Training DataFrames with `training_id`, features, `oracle_agent` |
| **Universe A** | `load_split`, `load_split_for_feature_set` |
| **Universe B** | `load_eval_parquet`, `eval_base_frame` |

### `routing/data/aliases.py`
| | |
|---|---|
| **IN** | Raw loader frames (`query_id`, `prob_*`, …) |
| **MID** | Rename to canonical columns (`training_id`, `p_*`) |
| **OUT** | Normalized DataFrame |

### `routing/features/columns.py`
| | |
|---|---|
| **IN** | Merged feature DataFrames |
| **MID** | Resolve `dim_*` / `emb_*` / `heur_*` / trust columns; path helpers |
| **OUT** | `router_feature_cols()`, `overall_complexity()` |
| **Universe** | Both |

### `routing/features/eval_builder.py`
| | |
|---|---|
| **IN** | Eval base frame, decomposer cache, QCE feature parquets |
| **MID** | Build / merge eval-time features per dataset tag |
| **OUT** | Feature-ready eval DataFrame |
| **Universe B** | `build_eval_features`, `ensure_eval_features` |

### `routing/train/pipeline.py`
| | |
|---|---|
| **IN** | QCE train/val splits, feature column list |
| **MID** | Sklearn HGBM pipeline, sample weights, train/eval metrics |
| **OUT** | Fitted pipeline, `EvalResult`, training baselines B0 |
| **Universe A** | `train_router`, `evaluate`, `fit_router` |

### `routing/train/experiments.py`
| | |
|---|---|
| **IN** | QCE splits, grid specs, CLI args |
| **MID** | Ablations, HGBM tune, seed sweep, LODO, router CLI `main()` |
| **OUT** | Experiment CSVs, tuned joblibs |
| **Universe A** | Primary training CLI (`research-route train`, …) |

### `routing/infer/predict.py`
| | |
|---|---|
| **IN** | Feature DataFrame, trained `.joblib` |
| **MID** | Load artifact, batch predict, rank agents |
| **OUT** | Routed frame (`router_pred`, `p_*`, `max_prob`, `assigned_agent`) |
| **Universe** | Both |

### `routing/runtime.py`
| | |
|---|---|
| **IN** | Loaded router, query row |
| **MID** | `RuntimeRouter` single-query inference; agent cascade execution |
| **OUT** | Agent assignment + optional multi-agent run |
| **Universe** | Both (orchestrator uses this at execution time) |

### `routing/router.py` (shim)
| | |
|---|---|
| **IN** | Same as submodules above |
| **MID** | Re-export stable public API for backward compatibility |
| **OUT** | All symbols previously imported from `routing.router` |
| **Deps** | `routing.data`, `routing.features`, `routing.train`, `routing.infer`, `routing.runtime` |

### `routing/soft_train.py`
| | |
|---|---|
| **IN** | QCE splits with `p_*` columns, optional `baselines_all.csv` |
| **MID** | Soft-KL row expansion, train HGBM/logreg, seed sweep, 4-way regret comparison |
| **OUT** | `models/router/graph_main/hgbm_cvec5_emb_soft_kl.joblib`, metrics CSV/JSON |
| **Deps** | `routing.baselines`, `routing.router`, `routing.analysis`, `soft_labels` |
| **Universe** | **A only** (internal splits) |

### `routing/baselines.py` — Step 2 classifier baselines
| | |
|---|---|
| **IN** | Internal QCE splits, `oracle_results1.csv`, frozen `LEGACY_HARD_HGBM_PATH` |
| **MID** | Train/eval logreg & LGBM on `cvec5_emb`; compare test utility regret |
| **OUT** | `results/experiments/classifier_baselines_cvec5_emb/baselines_all.csv` |
| **Deps** | `routing.analysis`, `routing.router`, `soft_labels` |
| **Universe** | **A only** — NOT live eval always-agent runs |

### `routing/selective.py`
| | |
|---|---|
| **IN** | Cheap/full router joblibs, QCE split or eval base frame, optional feature parquets |
| **MID** | Gate on `max_prob_emb`; partial vs full decompose; τ sweep |
| **OUT** | Gated routes (`selective_path`, `used_decompose`), `recommended_tau.json` |
| **Deps** | `routing.analysis` (oracle matrix), `routing.router`, `qce.*`, embed script |
| **Universe A** | `route_selective_split`, τ sweep on internal test |
| **Universe B** | `selective_build_and_route` via orchestrator `--selective-qce` |

---

## 4. Routing — eval & scoring

### `routing/analysis.py`
| | |
|---|---|
| **IN** | QCE split via `load_split`, `oracle_results1.csv`, router `.joblib` |
| **MID** | Outcome matrix · attach router · metrics · sweeps · plots · discussion slices |
| **OUT** | PNG, CSV, JSON under experiment dirs; `run_full_analysis` summaries |
| **Deps** | `routing.router`, `soft_labels.DEFAULT_UTILITY_LAMBDA` |
| **Universe A** | Primary consumer — `baseline_strategies`, internal regret, calibration |
| **Key fn** | `oracle_outcome_matrix` ← **internal outcome source** |

### `routing/oracle_bounds.py`
| | |
|---|---|
| **IN** | Live always-agent parquets under `LIVE_EVAL_BASELINE_ROOT`, eval_samples count via `load_eval_parquet` |
| **MID** | Join 4 agents per query; compute always/agent, Average, Best Agent, Oracle utility |
| **OUT** | `results/reports/live_eval/oracle_bounds.csv`, long CSV, per-query CSVs |
| **Deps** | `routing.benchmark`, `routing.router.load_eval_parquet`, `soft_labels` |
| **Universe** | **B only** — never reads `oracle_results1.csv` |

### `routing/score_routes.py`
| | |
|---|---|
| **IN** | Routes parquet (`training_id`, `router_pred`); **either** `split` **or** `dataset` |
| **MID** | Join routes to outcomes; compute EM, mUSD, utility regret |
| **OUT** | Scored DataFrame, summary dict (`outcome_source`: `"oracle"` \| `"live_baselines"`) |
| **Deps** | `routing.analysis`, `routing.oracle_bounds` (lazy for live) |
| **Universe** | **Both** — dispatch is the critical branch |

### `routing/benchmark.py`
| | |
|---|---|
| **IN** | `datasets/eval_samples/*.parquet`, production router joblib |
| **MID** | Route all eval datasets; optional score vs live baselines; launch orchestrator subprocess |
| **OUT** | `{dataset}_routes.parquet`, `router_distribution.csv/md`, orchestrator results |
| **Deps** | `routing.router`, `routing.score_routes`, `routing.oracle_bounds` (oracle-bounds subcmd) |
| **Universe** | **B only** for routing; subcommand `oracle-bounds` → universe B bounds |

### `routing/figures.py`
| | |
|---|---|
| **IN** | `baselines_all.csv`, soft-KL metrics CSVs |
| **MID** | Build paper figures (4-way regret bar chart) |
| **OUT** | PNG under `results/experiments/soft_kl_cvec5_emb/figures/` |
| **Universe** | **A** (internal test regret figure) |

---

## 5. Orchestrator

Phase 3 split the former ~980-line `orchestrator/pipeline.py`.  
Archived monolith: `archive/refactor_2026/orchestrator/pipeline.py`.

### `orchestrator/load.py`
| | |
|---|---|
| **IN** | Routes parquet or eval/QCE frame paths |
| **MID** | Load and normalize route/eval frames |
| **OUT** | Execution-ready DataFrame |

### `orchestrator/persist.py`
| | |
|---|---|
| **IN** | Pipeline results rows, checkpoints |
| **MID** | JSONL append, parquet writes, checkpoint resume |
| **OUT** | `pipeline_results.parquet`, `.jsonl`, checkpoints |

### `orchestrator/route_stage.py`
| | |
|---|---|
| **IN** | Eval/QCE frame, selective QCE config |
| **MID** | Route or selective-gate; merge route columns |
| **OUT** | Routed frame for execution stage |

### `orchestrator/run_manifest.py`
| | |
|---|---|
| **IN** | Run metadata, paths |
| **MID** | Write `manifest.json` (legacy + new layout) |
| **OUT** | Reproducibility manifest beside results |

### `orchestrator/pipeline.py` (thin runner)
| | |
|---|---|
| **IN** | **One of:** `--routes-path` (pre-routed) · `--split` (QCE) · `--dataset` (eval_samples) |
| **MID** | Compose load → route → execute → grade → persist → optional score_lookup |
| **OUT** | `pipeline_results.parquet`, `.jsonl`, `.summary.json`, optional `.lookup_score.json` |
| **Deps** | `routing.*`, `agent.*`, `evaluator.grade`, orchestrator submodules |
| **Universe A** | `--split train|val|test`; score_lookup uses `score_split` |
| **Universe B** | `--dataset gaia`; `--routes-path`; live baselines under `graph_main_tuned/` |
| **Entry** | `research-pipeline` console script |

### `orchestrator/__init__.py`
| | |
|---|---|
| **IN** | None |
| **MID** | Lazy export `run_pipeline` |
| **OUT** | Public API |

---

## 6. QCE (Phase 1 features)

### `qce/decompose.py`
| | |
|---|---|
| **IN** | Query rows (CSV/parquet), OpenAI API |
| **MID** | LLM plan decomposition |
| **OUT** | `datasets/decomposer_cache/qce_{tag}_plans.jsonl` |
| **Universe** | Both — tag = split name (A) or dataset name (B) |

### `qce/graph.py`
| | |
|---|---|
| **IN** | Decomposer cache JSONL |
| **MID** | Build procedure DAG (NetworkX) |
| **OUT** | In-memory graph structures → consumed by complexity |

### `qce/complexity.py`
| | |
|---|---|
| **IN** | Plans + DAG, `train_norm.json` |
| **MID** | Compute C(Q) vector (`dim_*`) + scalar `overall` |
| **OUT** | `datasets/qce_features/complexity_record_{tag}.parquet` |
| **Universe** | Both |

### `qce/query_input.py`, `qce/corpus_io.py`
| | |
|---|---|
| **IN** | Raw corpus rows |
| **MID** | Normalize row schema for decompose |
| **OUT** | `QueryIn` objects |

---

## 7. Agent & evaluation

### `agent/registry.py`
| | |
|---|---|
| **IN** | `strategy_name`, `query`, `model`, `dataset`, row kwargs |
| **MID** | Dispatch to strategy `run()` |
| **OUT** | `AgentResponse` (answer, cost, latency, …) |
| **Note** | 6 strategies; router trains on 4 (`raw`, `cot`, `react`, `multiagent`) |

### `agent/{raw,cot,react,multiagent,...}.py`
| | |
|---|---|
| **IN** | Query, model, dataset-specific kwargs (context, attachments) |
| **MID** | LLM calls + optional tools |
| **OUT** | `AgentResponse` |

### `agent/dataset_profile.py`, `agent/episode_context.py`
| | |
|---|---|
| **IN** | DataFrame row |
| **MID** | Build agent kwargs (Hotpot/MuSiQue context, GAIA metadata) |
| **OUT** | kwargs dict for `run_strategy` |

### `evaluator/grade.py`, `evaluator/parse.py`
| | |
|---|---|
| **IN** | Predicted vs expected answer, dataset name |
| **MID** | Dataset-specific normalization + EM |
| **OUT** | `bool` / parsed answer dict |

---

## 8. Labels & data scripts

### `src/utils/soft_labels.py`
| | |
|---|---|
| **IN** | `oracle_results*.csv`, QCE split CSVs |
| **MID** | Soft `p_*` from correct agents; utility argmax `oracle_agent`; update splits |
| **OUT** | Updated `qce_train.csv`, etc. with `p_*`, `oracle_agent` |
| **Universe** | **A only** — defines training supervision |

### `src/utils/supervision_labels.py`
| | |
|---|---|
| **IN** | Multi-agent benchmark checkpoint JSONL |
| **MID** | Expand to long-form strategy outcomes |
| **OUT** | Rows for building/exporting `oracle_results.csv` |
| **Universe** | **A** — feeds oracle CSV creation |

### `src/utils/train_samples.py`, `src/utils/stratified_sample.py`
| | |
|---|---|
| **IN** | Processed benchmark parquets |
| **MID** | Stratified sampling |
| **OUT** | Sampled DataFrames |

### `scripts/create_train_samples.py`
| | |
|---|---|
| **IN** | `datasets/processed/{math,hotpot,musique}/` |
| **MID** | 500-row stratified samples |
| **OUT** | `datasets/train_samples/*.parquet` → feeds QCE pool (universe A) |

### `scripts/create_eval_samples.py`
| | |
|---|---|
| **IN** | Processed test/val splits |
| **MID** | ~200-row eval samples + full GAIA val |
| **OUT** | `datasets/eval_samples/*.parquet` → universe B |

### `scripts/download.py`
| | |
|---|---|
| **IN** | Dataset name, `configs/datasets.yaml` |
| **MID** | Download + normalize |
| **OUT** | `datasets/processed/`, `datasets/raw/` |

### `src/data/registry.py` + `*_loader.py`
| | |
|---|---|
| **IN** | HuggingFace / local paths |
| **MID** | Load benchmark into common schema |
| **OUT** | Processed parquets |

---

## 9. Prompts

### `prompts/prompts_core.py`, `prompts/prompts.py`, `prompts/qce_decompose.py`
| | |
|---|---|
| **IN** | None (templates) |
| **MID** | Prompt string builders |
| **OUT** | Messages for agents / decompose LLM |

---

## 10. Dependency edges (refactor-critical)

```
soft_labels ──► qce split CSVs (oracle_agent, p_*)
                    │
                    ▼
router / soft_train ◄── analysis.oracle_outcome_matrix ◄── oracle_results1.csv
                    │
score_routes ──split──► outcomes_from_oracle (universe A)
             ──dataset──► outcomes_from_baselines ──► oracle_bounds ──► live baseline parquets (universe B)

benchmark (B) ──route-dist──► routes parquets
            ──score_routed_eval──► needs live baselines already on disk

orchestrator ──execute──► pipeline_results (creates live baselines when run as always-{agent})
             ──route-only + score_lookup──► score_routed (A or B dispatch)
```

---

## 11. Disk paths quick map (both universes)

| Role | Universe A (internal) | Universe B (live eval) |
|------|----------------------|----------------------|
| Queries | `train_samples/v1/qce_*.csv` | `eval_samples/*.parquet` |
| Features | `qce_features/*_{split}.parquet` | `qce_features/{tag}/` |
| Outcomes | `train_samples/v1/oracle_results1.csv` | `orchestrator/graph_main_tuned/.../baseline_{agent}/` |
| Routes | analysis output / split route-only | `experiments/*/routes/{ds}_routes.parquet` |
| Metrics | `analyze --split test` | `score-routes --dataset`, Table 5, oracle_bounds |
| Models | same `models/router/graph_main/` | same production router |

---

## 12. Refactor rules (from this catalog)

1. **Never** point universe A scoring at `LIVE_EVAL_BASELINE_ROOT` or vice versa.
2. **`outcomes.py` extraction** must expose `from_oracle_csv()` and `from_live_baselines()` as explicit functions.
3. **`baseline` in filenames** — use prefix: `internal_`, `live_`, `classifier_`, `training_b0_`.
4. **Orchestrator** pre-routed files (`*_baseline_{agent}_routes.parquet`) are universe B **route inputs**, not outcomes.
5. **Table 5 / `oracle_bounds_long.csv`** depend entirely on universe B physical baselines — do not regenerate from oracle CSV.
