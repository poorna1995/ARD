# D-AAR thesis freeze (2026-06)

Historical experiments archived here. The **active thesis system** is the decomposed
`cvec5` pipeline only.

## Supervisor-aligned pipeline (frozen thesis)

```text
Query → QCE → G(q), φ(q), emb(q)
  → Gate 1a P(solvable)
  → [abstain | Gate 1b ∥ Gate 2 → Gate 3 U=ŝ−λĉ → execute â]
```

Headline val: **θ_s_joint = 0.05**, **λ = 0** (regret-optimal). Deployment: λ>0 via
`datasets/daar/routing/utility_lambda_pareto_val.json`.

## One method, three ablations

| Version | Location | Role |
|---------|----------|------|
| **Decomposed BEST (`cvec5`)** | `models/daar/`, `scripts/train_gate1*.py`, `daar/` | **Final method** |
| P0 | `p0/` | Negative ablation (per-agent binary heads) |
| P1 | `p1/` | Intermediate (utility softmax rank — superseded) |
| Exploratory | `exploratory/` | θ/λ sweeps, Gate1b grids, ablation models, audits |

## Active project (keep — reproduce final thesis)

### Docs
- `docs/D-AAR.md` (single canonical doc)

### Frozen models (`models/daar/`)
- `solvability_gate1a.joblib` + `solvability_gate1a_manifest.json`
- `solvability_gate1b_soft_kl.joblib` + `solvability_gate1b_manifest.json`
- `cost_hand_manifest.json`

### Final routing JSONs (`datasets/daar/routing/`)
- `thesis_final_experiments.json`
- `per_dataset_regret_val.json`
- `eval_samples_decomposed.json`
- `utility_lambda_pareto_val.json`
- `gaia_example_trace.json`
- `gate1a_cvec5_ablation.json` (result; model in `exploratory/models/`)
- `gate1b_sweep_results.json`
- `decomposed_routing_manifest.json`

### Repro scripts (`scripts/`)
- Build: `export_daar_oracle.py`, `build_daar_*`, `decompose_qce_plans.py`,
  `build_daar_trace_skeleton.py`, `simulate_daar_cost_hand.py`
- Train: `train_gate1a_solvability.py`, `train_gate1b_success_soft.py`
- Eval: `route_daar.py`, `run_thesis_final_experiments.py`, `eval_*`,
  `exploratory/scripts/ablate_gate1a_cvec5.py`, `build_gaia_worked_example.py`, `build_eval_cost_hand.py`

## Moved to archive (2026-06 cleanup)

| From | To | Why |
|------|-----|-----|
| `models/daar/solvability_gate1a_cvec5.*` | `exploratory/models/` | Gate 1a ablation (tie at θ=0.05; not frozen deploy) |
| `datasets/daar/routing/lambda_sweep_val_decomposed.csv` | `exploratory/routing/` | Duplicate of manifest sweep |
| `datasets/daar/routing/daar_decomposed_val_predictions.parquet` | `exploratory/routing/` | Regenerable via `route_daar.py` |
| `scripts/rerun_self_consistency.py` | `exploratory/scripts/legacy/` | Non-thesis experiment |
| `scripts/derive_ac_features.py` | `exploratory/scripts/legacy/` | Legacy AC features (D-AAR uses QCE cvec5) |
| `scripts/build_text_complexity_features.py` | `exploratory/scripts/legacy/` | Superseded by QCE |
| `scripts/export_offline_dataset_master.py` | `exploratory/scripts/legacy/` | Legacy data export |

### Removed (not reproduced)
- `gate1b_tfidf_svd` ablation (undone)
- `robustness_seed_eval` (HGBM deterministic on pool)

## Reproduce archived runs

```bash
# P0 / P1 ablations
uv run python archive/daar_thesis_freeze/p0/scripts/train_daar_solvability.py
uv run python archive/daar_thesis_freeze/p1/scripts/train_daar_solvability_p1.py

# Gate 1a cvec5 ablation (writes to exploratory/models/)
uv run python archive/daar_thesis_freeze/exploratory/scripts/ablate_gate1a_cvec5.py

# Headline decomposed val
uv run python scripts/route_daar.py --theta-joint-regret
uv run python scripts/run_thesis_final_experiments.py
```

P0/P1 frozen test numbers: `p0/routing/`, `p1/routing/`.
