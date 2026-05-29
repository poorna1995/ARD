# dim5/dim7 Secondary Manifest

This manifest records files and artifacts related to the secondary
`dim5`/`dim7` complexity experiments.

## Feature definitions

- `qce/graph.py`
  - `DIM_COLS` (`dim_*`, main 5-dim v3.0)
  - `DIM5_LEGACY_COLS` (`dim_legacy_*`, secondary legacy 5-dim)
  - `DIM7_COLS` (`dim7_*`, 7-dim)
- `qce/complexity.py`
  - `C_VECTOR_COLS` and router feature extraction helpers

## Router training / inference integration

- `routing/router.py`
  - feature-set handling for `dim_*` and `dim7_*`

## Architecture notes

- `docs/qce_router_architecture.md`

## Router artifacts (current repo)

- `models/router/hgbm_graph_emb_balanced.*`
- `models/router/hgbm_graph_emb_v2_balanced.*`
- `models/router/G*_hgbm_graph*.*`
- `models/router/G*_mlp_graph*.*`
- `models/router/T_*.*`

## Historical/archived data references

- `old/qce_features/*`
- `old/archive1/hotpot_train_ids_20260527/*qce*`

## Secondary experiment commands (reference)

- Train:
  - `uv run python -m routing train --save`
- Route/eval:
  - `uv run python orchestrator/pipeline.py --dataset <name> --build-features --grade`
- Analysis:
  - `uv run python -m routing.analysis ...`

> Note: keep this track as secondary. Re-promote to main only after rerunning
> and validating results end-to-end.

