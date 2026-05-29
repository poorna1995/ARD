# Secondary Experiment Track: dim5 + dim7

This folder isolates the `dim5` and `dim7` query-complexity experiments as a
secondary track. The goal is to keep the main project narrative clean while
preserving this work for later reuse.

## Scope

- Keep `dim5` and `dim7` as optional/secondary experiments.
- Do not treat these experiments as the default production path.
- Preserve reproducibility (features, router variants, and analysis commands).

## Core code locations (source of truth)

- `qce/graph.py`
- `qce/complexity.py`
- `routing/router.py`
- `docs/qce_router_architecture.md`

## Related model artifacts

See `manifest.md` in this folder for the exact artifact list and role.

## Suggested usage

Use this track only when explicitly running secondary analyses:

1. Build QCE features (`dim_*` and `dim7_*` available in generated records).
2. Train router variant tied to `dim5` or `dim7` experiment settings.
3. Evaluate against static baselines and report as supplementary/ablation.

## Reporting guidance

- Main paper: keep primary claims on stable main pipeline.
- Secondary section/appendix: report dim5/dim7 ablations and diagnostics.
- Avoid mixing secondary-track conclusions into main claims unless revalidated.

