# Architecture & refactor plan (saved from design session)

**Status:** Planning only — implement on a **separate branch** after oracle benchmark experiments finish.  
**Do not refactor on the branch running** `run_training_benchmark.py` merges on `logs/benchmark_runs/combined_raw/`.

---

## What you are running now (do not break)

| Active work | Entry point | Output |
|-------------|-------------|--------|
| Oracle benchmark (6 agents) | `scripts/run_training_benchmark.py` | `checkpoint.jsonl`, `results_runs.parquet` |
| Oracle labels / analysis | `notebooks/agent_oracle(math+hotqa.ipynb` | `oracle_results.csv`, `train_labels.csv` |
| Merge in progress | `--merge-agents --agents react multiagent` | patches checkpoint per `training_id` |

**Frozen contracts until refactor branch:**

- `run(query, model, dataset, **kwargs) -> AgentResponse`
- `evaluator.parse` / `evaluator.match` / `is_correct` semantics
- Checkpoint JSON shape + merge/resume modes
- `results_runs.parquet` columns used by the notebook

---

## Four research tracks (do not merge into one script)

```
Track 0  PREP     src/data, src/utils/train_samples → combined_raw.parquet
Track 1  ORACLE   agent/registry (6 strategies) + run_training_benchmark
Track 2  LABELS   src/utils/supervision_labels + notebook → oracle CSVs
Track 3  QCE      difficulty/feature_measure + routing/router (4 agents)
Track 4  COMPARE  baselines/ (gitignored; only routellm/ on disk today)
```

**“Baseline” overload:**

- **Agent strategies** = raw, cot, react, … (`agent/registry.py`) — your main methods.
- **`baselines/` package** = separate eval framework (mostly missing; tests in `tests/test_integration_baseline.py`).
- **RouteLLM** = `baselines/routellm/routellm.py` — strong/weak **model** routing (external), not QCE.

---

## Target layout (pipeline-first + connection layer)

```
research_work/
├── configs/datasets.yaml
├── datasets/                    # artifacts only
├── prep/                        # loaders, sampling, math_latex (from src/)
├── agent/                       # KEEP package name
│   ├── config.py, base.py, registry.py, profiles.py
│   ├── strategies/              # raw, cot, react, multiagent, sc, debate
│   └── tools/                   # stay inside agent/
├── prompts/                     # package; split from prompts.py later
├── grading/                     # from evaluator/ (+ shims)
├── benchmark/                   # checkpoint, executor, exports, modes
├── labels/                      # supervision + oracle CSV export
├── qce/                         # from difficulty/feature_measure.py
├── routing/                     # router.py + pipeline (from orchestrator/)
├── baselines/                   # comparison lane only — not merged into agent/
│   └── routellm/
├── pipeline/                    # ★ glue: context, artifacts, stages, recipes
│   ├── context.py, artifacts.py, records.py, composer.py
│   └── stages/ + recipes/
├── scripts/                     # thin CLIs → pipeline recipes
└── notebooks/
```

**Connection model:**

1. **QuestionRecord** — shared row: `training_id`, `query`, `reference`, `dataset_source`, `metadata`, optional `strategies`, `qce`, `routed_agent`, supervision fields.
2. **ArtifactStore** — named slots under `run_dir`: `corpus`, `checkpoint`, `runs`, `labels`, `qce_scores`, `routed`, …
3. **Recipe** — ordered stages, e.g. `oracle_full`, `oracle_merge_agents`, `oracle_labels_only`, `qce_route_eval`, `oracle_plus_routellm`.

---

## Example recipes (when pipeline/ exists)

| Recipe | Stages |
|--------|--------|
| `oracle_full` | load_corpus → run_oracle_benchmark → export_runs → build_labels |
| `oracle_merge_agents` | run_oracle_benchmark (mode=merge) → export_runs |
| `oracle_labels_only` | export_runs → build_labels → export_oracle_csv |
| `qce_route_eval` | load_corpus → fit_qce → score_qce → route_queries → [run_routed_agents] |
| `oracle_plus_routellm` | export_runs + run_routellm → join_experiments |

---

## Safe refactor phases (separate branch)

| Phase | Work | Safe during live benchmark? |
|-------|------|---------------------------|
| 0 | Snapshot: prompt hashes, checkpoint samples, run evaluator tests | Yes |
| 1 | Extract `benchmark/checkpoint.py` + `exports.py`; script imports only | Yes |
| 2 | `benchmark/executor.py`, `modes.py`; thin CLI | Yes |
| 3 | `pipeline/context`, `artifacts`, `records` | Yes |
| 4 | `labels/` + unified checkpoint loader | Yes |
| 5 | `scripts/run_pipeline.py` + recipes | After merge stable |
| 6 | `qce/` + routing stages | Yes |
| 7 | `agent/strategies/` move + shims | Careful |
| 8 | `prompts/` package split + shim | Own PR |
| 9 | `grading/` rename + shim | Own PR |
| 10 | cot/sc: `strategy=` + `_core_response` | Behavior PR |
| 11 | Remove shims, quarantine `_legacy/` | Last |

**Do not:** rename `agent` → `agents`; move `tools/` out of `agent/`; delete commented code before shims work.

---

## Known gaps (fix on refactor branch, not urgent)

- `cot` / `self_consistency` missing `strategy=` in `normalize_agent_config`.
- `src/utils/supervision_labels.load_checkpoint_jsonl` simpler than benchmark loader — unify on `benchmark/checkpoint.py`.
- QCE: router must use `evaluate()`, not `analyze()` (threshold drift).
- `baselines/` gitignored; `routellm.py` imports missing `src.loaders`, `baselines.prompt_resolver`.
- `tests/test_integration_baseline.py` imports absent `baselines.*`.
- `run_agents_shared_config.py` references missing `agent.muliatgent1`.
- Root `run.py` broken (`src.complexity`).

---

## Import direction

```
pipeline → benchmark, labels, qce, routing, agent.registry
agent → prompts, grading, agent.tools (never pipeline)
grading → data.math_latex only
baselines → prep loaders + own prompts (or adapter to prompts/)
```

---

## When experiments finish

1. Branch from clean main: e.g. `refactor/pipeline-structure`.
2. Phase 1 only first PR: checkpoint_io extract, zero behavior change.
3. Keep `scripts/run_training_benchmark.py` as wrapper until recipes proven.
4. Re-run: evaluator tests, `agent/test_dataset_profile`, `--limit 1` benchmark, notebook read of `results_runs.parquet`.

---

*Generated for handoff; implementation not started on experiment branch.*
