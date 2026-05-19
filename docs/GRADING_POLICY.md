# Grading policy (implemented)

This matches the decisions you recorded for oracle / routing labels (`evaluator/dataset_normalize.py`).

## HotpotQA & MuSiQue

- **Metric:** SQuAD-style **token F1** (multiset overlap), same normalization chain as HotpotQA `normalize_answer` plus the existing hyphen / apostrophe pre-pass.
- **Threshold:** **F1 ≥ 0.5** and **precision ≥ 0.5** and **recall ≥ 0.5** (blocks extra wrong tokens when gold is short). Normalized exact match always passes.
- **Multi-entity guard:** If normalized gold contains the substring ` and ` (e.g. two people), require **F1 ≥ 0.62** so `train_0586`-style “first name only” answers (~0.57) stay **wrong**, while typical single-entity answers (e.g. stage name vs longer legal string without ` and `) can still pass at **0.5**.
- **Not implemented:** blanket substring acceptance, geo relaxation (e.g. Mexico vs Tamaulipas), date off-by-one.

## MATH

1. **Surface normalize** before Hendrycks `is_equiv`: thousands commas, `sqrt(` → `\sqrt{…}`, `*` between digit and `\`, `100*pi` → `100\pi`, bare `pi` → `\pi` (word-boundary safe).
2. Then **`is_equiv`** (upstream `math_equivalence`).
3. Then **base-suffix** integer equivalence (`1` vs `1_6`, etc.) on raw strings as before.
- **Not implemented:** float ≈ symbolic (e.g. `9.87` vs `π²`) — would need a separate symbolic/numeric layer.

## GAIA

Unchanged: leaderboard `question_scorer` behavior.

## MMLU-Pro

Unchanged: letter extraction + match.

## What you explicitly deferred

- **Prose dumps:** still graded on full `predicted_answer`; fix via **extraction** or prompts, not by accepting long text as correct.
- **Substring / generic phrases:** no special accept path.

Recompute labels after any change:

```bash
uv run python scripts/regrade_train_labels.py
```
