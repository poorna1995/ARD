# QCE + Agent Router — System Flow

**Start here:** open [figures/qce_router_architecture.svg](figures/qce_router_architecture.svg) (three numbered phases, left-to-right).

The system does one thing in three steps:

1. **Turn each query into a feature row** (numbers describing how hard the query is + a short embedding).
2. **Train a model** on past queries to predict which agent works best.
3. **At runtime**, build features for a new query, run the model, execute the chosen agent.

---

## Phase 1 — Build features (QCE)

**Goal:** One row of numbers per query.

| Step | What happens | Code |
|------|----------------|------|
| 1 | You have a **query** string and **dataset** name (GAIA, Hotpot, …) | `orchestrator/pipeline.py` |
| 2 | **LLM decompose** writes a plan: subtasks, tools, dependencies | `qce/decompose.py` → `datasets/decomposer_cache/*.jsonl` |
| 3 | Plan becomes a **procedure DAG** (nodes = subtasks, edges = depends_on) | `qce/graph.py` |
| 4 | DAG + plan text → **C(Q)** main 5 values: `dim_structural`, `dim_reasoning`, `dim_evidence`, `dim_tool`, `dim_coordination_uncertainty` | `qce/complexity.py` |
| — | In parallel: query text → embedding → **PCA-16** → `emb_*` columns | `scripts/build_query_embeddings.py` |
| 5 | **Merge** C(Q) + emb_* + dataset → one **feature row** | `routing/router.py` `merge_feature_tables` |

**On disk:** `datasets/qce_features/complexity_record_{split}.parquet` and `query_embeddings_{split}.parquet`.

---

## Phase 2 — Train router (offline)

**Goal:** Learn `query features → best agent`.

| Step | What happens |
|------|----------------|
| 6 | Build a **training table**: feature row + **hard** `oracle_agent` (utility argmax) and/or **soft** `p_*` (cost-aware mass over *correct* agents only) |
| 7 | **Soft-train** → KL / soft CE on `p_*` with `dim_*` + `emb_*` (`cvec5_emb`) |
| 8 | Save role-specific checkpoints (see table below); **`ROUTER_MODEL_PATH`** → `hgbm_cvec5_emb_soft_kl.joblib` |

**Router roles (internal test, λ=25, train n=808):**

| Role | Model | Artifact | Test regret |
|------|-------|----------|-------------|
| **Primary** | Soft HGBM | `hgbm_cvec5_emb_soft_kl.joblib` | 0.192 |
| **Secondary** | Soft Logistic | `logreg_cvec5_emb_soft_kl.joblib` | 0.199 |
| **Legacy** | Hard HGBM | `hgbm_cvec5_emb_default.joblib` | 0.361 |

**Model output:** For each query, four probabilities that sum to 1:

`p_raw`, `p_cot`, `p_react`, `p_multiagent`

**Train commands:** `soft-train --classifier hgbm --save` (primary); `soft-train --classifier logreg --save` (secondary); `routing train --feature-set cvec5_emb --save` (legacy hard HGBM only)

---

## Phase 3 — Inference (runtime)

**Goal:** Pick one agent for a new query and run it.

| Step | What happens |
|------|----------------|
| 9 | New **eval query** (e.g. GAIA benchmark) |
| 10 | Run **Phase 1** again (`--build-features` if not cached) |
| 11 | Load `.joblib`, **predict_proba**, take **argmax** → `assigned_agent` |
| 12 | **Run** that agent via `agent/registry.py` |
| 13 | **Grade** answer vs `expected_answer` |

**Example:** If `p_react = 0.62` is highest → run ReAct.

**Optional:** `cascade_on_fail` tries the 2nd/3rd agent if the first returns empty or errors.

**Run command:** `uv run python orchestrator/pipeline.py --dataset gaia --build-features --grade`

---

## Simple diagram (Mermaid)

```mermaid
flowchart TB
  subgraph P1["Phase 1 — Features"]
    Q[Query] --> D[LLM Decompose]
    D --> G[Procedure DAG]
    G --> C[C(Q) vector]
    Q --> E[Query embedding]
    C --> F[Feature row]
    E --> F
  end

  subgraph P2["Phase 2 — Train"]
    F --> T[Training table + oracle labels]
    T --> M[HGBM model .joblib]
  end

  subgraph P3["Phase 3 — Inference"]
    Q2[New query] --> F2[Feature row]
    F2 --> R[Router argmax]
    M --> R
    R --> A[Run one agent]
    A --> AN[Answer + grade]
  end

  P1 --> P2
  P2 --> P3
```

---

## vs RouterGNN (reference paper figure)

| RouterGNN | This project |
|-----------|--------------|
| Entity knowledge graph | **Procedure DAG** from LLM plan |
| GNN message passing | **5-dim C(Q)** + embedding (tabular) |
| KL on query–agent edges | **Cross-entropy** on `oracle_agent` |
| Graph at inference | **Precomputed features** + HGBM |

---

## Module map

| Phase | Modules |
|-------|---------|
| 1 | `qce/decompose.py`, `qce/graph.py`, `qce/complexity.py` |
| 2 | `routing/router.py` |
| 3 | `orchestrator/pipeline.py`, `agent/registry.py` |
