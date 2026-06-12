# Refactor batch: refactor_2026

| Date | Change | Archive path | Replaced by |
|------|--------|--------------|-------------|
| 2026-05-30 | Phase 0 — safety net | _(none)_ | tests/, routing/verify.py |
| 2026-05-30 | Phase 1 — config + heuristics | archive/.../heuristic_cols_old_import.md | config/, routing/features/heuristics.py |
| 2026-05-30 | Phase 2 — dedupe eval logic | _(inline removed from analysis/router/selective/oracle_bounds/score_routes)_ | routing/eval/outcomes.py, metrics.py · routing/features/eval_builder.py · routing/datasets.py |
| 2026-05-30 | Phase 3 — split router + orchestrator monoliths | archive/refactor_2026/routing/router.py · archive/refactor_2026/orchestrator/pipeline.py | routing/data/splits.py · routing/features/columns.py · routing/train/pipeline.py · routing/train/experiments.py · routing/infer/predict.py · routing/runtime.py · routing/router.py (shim) · orchestrator/load.py · orchestrator/persist.py · orchestrator/route_stage.py · orchestrator/pipeline.py (thin) |
| 2026-05-30 | Phase 4 — package + Docker + unified CLI | _(pyproject `package=false` in git history)_ | `routing/cli.py` · `research-route` / `research-pipeline` · `deploy/docker-compose.prod.yml` |
| 2026-05-30 | Phase 5 — opt-in data layout | _(symlinks via setup script)_ | `config/paths.py` · `routing/data/aliases.py` · `orchestrator/run_manifest.py` · `scripts/setup_data_layout.py` |
| 2026-05-30 | Phase 6 — production hardening | _(none)_ | `config/logging.py` · `config/secrets.py` · `config/llm_retry.py` · `api/main.py` · `.github/workflows/ci.yml` · `README.md` |

| 2026-05-30 | Consolidate split modules → minimal file count | archive/refactor_2026/routing/split_modules/ · archive/refactor_2026/orchestrator/split_modules/ | `routing/router.py` (~1020 lines) · `routing/experiments.py` · `orchestrator/pipeline.py` (~1065 lines) |

| 2026-05-30 | Config layout — common / global_config / local | archive/refactor_2026/config/pre_move/ | `config/common.py` · `config/global_config/` · `config/local/` |

## Config layout (2026-05-30)

```
config/
  common.py              REPO_ROOT, USE_NEW_DATA_LAYOUT, load_yaml()
  global_config/
    paths.py              all disk paths (input + QCE + router + results)
    runtime.py              env, logging, secrets, llm_retry
  local/
    router.py               HGBM paths, feature sets, agents
    datasets.yaml           benchmark download config

Legacy shim: ``router/config.py`` only.


## INPUT package (2026-05-30)

```
input/
  paths.py              raw/processed/train/eval/labels path resolvers
  download.py           HF download + loader CLI
  layout.py             symlinks legacy → datasets/input/
  loaders/              base, registry, math, musique, gaia, hotpot, mmlu_pro
  samples/              stratified.py, train_samples.py, build_train.py, build_eval.py
  preprocess/           math_latex.py, normalize_math.py
  prompts/              prompts_core.py, prompts.py, qce_decompose.py
```

Legacy imports (`src.data.*`, `src.math_latex`, `src.utils.train_samples`) remain via shims.
Scripts (`scripts/download.py`, etc.) delegate to `input.*`.

Disk layout (opt-in ``RESEARCH_USE_NEW_PATHS=1``)::

    datasets/input/raw/{dataset}/
    datasets/input/processed/{dataset}/
    datasets/input/samples/eval/*.parquet
    datasets/input/samples/train/*.parquet
    datasets/input/labels/v1/

Setup::

    uv run python scripts/setup_data_layout.py
    uv run python -m input.download --dataset math
    uv run python -m input.samples.build_train


## Package layout (2026-05-30)

```
router/           router.py, config.py, experiments.py
eval/             score.py, benchmark.py, oracle_bounds.py, features.py
research/         analysis.py, soft_train.py, baselines.py, figures.py, selective.py
input/            paths.py, download.py, layout.py, loaders/, samples/, preprocess/
routing/          cli.py, verify.py, _compat.py, __init__.py, __main__.py (5 files — no subfolders)
```

Old `routing/*.py` backed up to `archive/refactor_2026/routing/pre_package_layout/`.

## Consolidation (2026-05-30)

Phase 3 split was reversed for a **minimal codebase** (same behavior, fewer files):

- **Router:** `routing/router.py` — load, features, train, infer, runtime (one module)
- **CLI/ablations:** `routing/experiments.py`
- **Orchestrator:** `orchestrator/pipeline.py` — load, persist, route, manifest, run

Removed from active tree: `routing/data/`, `routing/train/`, `routing/infer/`, `routing/runtime.py`, `routing/features/columns.py`, orchestrator submodules.

## Phase 3 (complete when verify + pytest pass)

- `routing/router.py` (1667 lines) → data / features / train / infer / runtime + compatibility shim
- `orchestrator/pipeline.py` (980 lines) → load / persist / route_stage + `run_pipeline` entrypoint
- Public imports via `routing.router` and `routing.__init__` unchanged

## Restore (Phase 3)

```bash
cp archive/refactor_2026/routing/router.py routing/router.py
cp archive/refactor_2026/orchestrator/pipeline.py orchestrator/pipeline.py
# remove routing/data/, routing/train/, routing/infer/, routing/runtime.py, routing/features/columns.py
# remove orchestrator/load.py, persist.py, route_stage.py
```

## Phase 4 (complete when verify + pytest pass)

- `pyproject.toml`: `package = true`, hatchling wheel, console scripts
- `routing/cli.py`: unified CLI; **default subcommand = verify**
- `python -m routing` and `research-route` equivalent; `python -m orchestrator` → pipeline
- Docker: installs project, `HEALTHCHECK research-route verify --skip-golden`
- Production model manifest fields on `hgbm_cvec5_emb_soft_kl.json` (`version`, `test_regret`, `train_n`)
- Removed `sys.path.insert` from orchestrator modules

## Restore (Phase 4)

Set `[tool.uv] package = false` and remove `[project.scripts]` / `[build-system]` from `pyproject.toml`; use `uv run python -m routing` only.

| 2026-05-30 | Phase 5 — opt-in data layout | _(none — symlinks only)_ | `config/paths.py` resolvers · `routing/data/aliases.py` · `orchestrator/run_manifest.py` · `scripts/setup_data_layout.py` |

## Phase 5 (complete when verify + pytest pass)

- `RESEARCH_USE_NEW_PATHS=1` switches dataset/model path resolvers in `config/paths.py`
- Symlinks: `datasets/input/`, `datasets/intermediate/`, `models/router/production/` → legacy files
- Column aliases: `query_id` → `training_id`, `prob_*` → `p_*` in loaders
- New orchestrator runs (new layout, no `--output_path`) → `results/runs/{date}_{experiment_id}/` + `manifest.json`
- `LIVE_EVAL_BASELINE_ROOT` unchanged (`results/orchestrator/graph_main_tuned`)

Setup symlinks::

    uv run python scripts/setup_data_layout.py

## Restore (Phase 5)

```bash
export RESEARCH_USE_NEW_PATHS=0   # default
# remove datasets/input/, datasets/intermediate/, models/router/production symlink if desired
```

| 2026-05-30 | Phase 6 — production hardening | _(none)_ | `config/logging.py` · `config/secrets.py` · `config/llm_retry.py` · `api/main.py` · `.github/workflows/ci.yml` · `README.md` |

## Phase 6 (complete when CI + pytest pass)

- Structured logging: `RESEARCH_LOG_JSON`, `RESEARCH_LOG_LEVEL`
- Secrets check in `research-route verify --require-secrets`
- LLM retry/rate-limit in `qce/decompose.py` via `RESEARCH_LLM_*`
- Run `manifest.json` written for **all** orchestrator runs (legacy + new layout)
- Optional FastAPI: `uv sync --group api` → `uvicorn api.main:app`
- CI: pytest + ruff + verify (legacy + new paths)

## Shim deprecation (Phase 4+)

Keep `routing/router.py` and `python -m routing` shims until **2026-06-30** (30 days after Phase 4).
After that window, migrate imports to `routing.infer`, `routing.train`, etc. and remove shims to `archive/`.

## Restore (Phase 6)

N/A (additive). Disable CI by removing `.github/workflows/ci.yml`.


- Added `routing/verify.py`, `tests/test_*.py`, `tests/golden/build_fixtures.py`
- No monolith moves in this batch

## Restore

N/A for Phase 0 (additive only).

## Verification run

```bash
uv run pytest tests/ -q
uv run python -m routing verify
```
