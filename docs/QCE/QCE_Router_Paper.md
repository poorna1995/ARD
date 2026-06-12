# Adaptive Agent Routing (AAR): Plan-Structure-Guided Cost-Soft Selection for LLM Agent Strategies

**Anonymous Authors**  
*Affiliation TBD*

*Compile LaTeX from `docs/`: `pdflatex QCE_Router_Paper_main` → `bibtex QCE_Router_Paper_main` → `pdflatex` ×2. Bibliography: `QCE_Router_Paper.bib`. In-text: `\citep{key}` (Author, Year), `\citet{key}` (Author (Year)).*

---

## Abstract

Deploying large language model (LLM) agents in practice requires choosing among heterogeneous reasoning strategies—direct prompting, chain-of-thought, tool-augmented ReAct, and multi-agent coordination—yet no single strategy is optimal across queries or benchmark families. Despite growing interest in adaptive routing, many systems rely on shallow query features and hard winner-take-all supervision, which discard complementarity among *correct* strategies that differ mainly in execution cost.

We propose **Adaptive Agent Routing (AAR)**, a pre-inference routing framework that (i) estimates procedural difficulty from an LLM-derived procedure directed acyclic graph (DAG) and a five-dimensional complexity vector, (ii) fuses procedural complexity with semantic query embeddings, and (iii) trains a router with **cost-soft** targets via Kullback–Leibler divergence, preserving mass over cheaper correct strategies.

Deploying LLM agents requires routing among heterogeneous strategies before execution. Existing routers rely on semantic features and hard labels, overlooking procedural demands and cheaper correct alternatives. **AAR** fuses planner-derived complexity with semantic embeddings and trains with **cost-soft supervision**. On a routing benchmark (*n*=100; λ=25), cost-soft cuts agent regret (0.361→0.192; EM 63%→82%); fused features reach lowest total regret (0.198), beating embedding-only, complexity-only, and heuristic-plus-embedding baselines. Complexity alone is insufficient but complementary. Transfer (*N*=965) is competitive but mixed (48.1% vs. 50.3% pooled EM).

---

## 1 Introduction

Large language models (LLMs) increasingly power agentic systems that perform multi-step reasoning, retrieval, tool use, and coordinated execution. As these agentic systems mature, they expose multiple **agent strategies**—raw direct prompting, chain-of-thought (CoT), ReAct-style tool use, and multi-agent coordination—each with distinct exact-match performance and inference costs for the same query.

Despite this growing diversity, deployed agentic systems typically apply a **single fixed agent strategy** to every query. Query demands are heterogeneous: some instances are solvable with lightweight prompting, while others require multi-step reasoning, retrieval, or multi-agent orchestration. A static **routing policy** can therefore misalign token cost with task requirements and weaken the **performance–cost trade-off**. On five executed workloads—**GAIA**, **HotpotQA**, **MuSiQue**, **MMLU-Pro**, and **MATH** (*N*=965)—the **best fixed agent strategy** varies by benchmark family (Figure 1): ReAct is strongest on four benchmarks and chain-of-thought on MATH, so no single static choice is optimal across heterogeneous queries.

**Figure 1.** Fixed-strategy exact match (%) on executed benchmark workloads (*n*=965).

| Workload | raw | cot | react | multiagent | Best fixed |
|----------|----:|----:|------:|-----------:|------------|
| GAIA | 4.2 | 6.1 | **26.1** | 13.3 | react |
| HotpotQA | 39.5 | 36.5 | **69.0** | 32.0 | react |
| MuSiQue | 2.5 | 2.5 | **36.5** | 25.5 | react |
| MMLU-Pro | 48.5 | 53.0 | **59.0** | 46.5 | react |
| MATH | 31.0 | **69.5** | 63.0 | 56.0 | cot |

Prior routing methods demonstrate that adaptive selection can improve efficiency. However, much of this literature targets *model routing*—selecting among strong and weak LLMs or API tiers—rather than routing among heterogeneous **agent strategies**, which differ in procedural structure as well as cost.

Routing decisions in related work are often based on semantic query representations, query–candidate interaction graphs, or latent difficulty estimates, and seldom represent the anticipated execution procedure implied by query decomposition. Semantically similar queries may nevertheless require substantially different execution procedures because of differences in dependency structure, reasoning depth, tool requirements, or coordination demands. When supervised routing uses a single utility-maximizing label, **hard supervision** also discards information when multiple agent strategies succeed at different costs.

Unlike prior routers that rely primarily on flat query embeddings, **AAR** incorporates planner-derived procedural complexity as an explicit signal for pre-inference **strategy selection** over 𝒜 = {raw, cot, react, multiagent}. In addition, AAR uses **cost-soft supervision** that preserves information from multiple successful agent strategies rather than enforcing a single winner. Procedural complexity provides information about anticipated operational requirements that is unavailable from semantic representations alone. **Consequently**, we formulate pre-inference routing as minimizing **utility regret** relative to the cost-penalized oracle *a*⋆(*q*) (§2).

**Research question. RQ1:** Given a query, can we estimate procedural complexity before any agent strategy runs and use that estimate to route the query to an appropriate agent strategy from 𝒜 so as to improve the performance–cost trade-off under single-strategy execution?

**Our solution.** To answer RQ1, we propose **AAR**, a pre-inference agent-strategy routing framework. The framework first estimates procedural complexity from planner-generated structures. It then combines complexity and semantic signals to form routing representations and learns a **routing policy** through cost-aware supervision derived from empirical agent outcomes. At **deployment**, AAR executes only the selected agent strategy in 𝒜, preserving the efficiency of single-strategy deployment. Architectural and algorithmic details appear in §3.

**Why it matters?** Procedural complexity estimated from planner-generated structures enables a **routing policy** that reserves `react` or `multiagent` for queries whose plans imply retrieval, tools, or coordination, while assigning `raw` or `cot` when appropriate. **Cost-soft supervision** further aligns training with deployment utility *U*ₐ(*q*) = *r*ₐ(*q*) − λ*c*ₐ(*q*) when multiple agent strategies in 𝒜 are correct at different costs. **Consequently**, deployed agentic systems can improve the cost-aware trade-off under single-strategy execution without executing all four agent strategies on every query.

**Contributions.**

1. We introduce a **complexity–semantic representation** for pre-inference routing over 𝒜, combining procedural complexity with semantic query embeddings.
2. We propose **cost-soft supervision** that preserves information from multiple successful agent strategies that hard labels discard.
3. We evaluate AAR on a stratified **routing benchmark** built from oracle four-strategy runs (HotpotQA, MuSiQue, MATH; train/val/test 808/100/100) and on benchmark transfer to GAIA, HotpotQA, MuSiQue, MMLU-Pro, and MATH, including representation and supervision ablations; cost-soft training with complexity–semantic features reduces utility regret relative to hard-label supervision and to embedding-only and heuristic baselines on the held-out routing benchmark (§4).

---

## 2 Problem Formulation

### 2.1 Pre-Inference Agent-Strategy Routing

Given query *q*, select an **agent strategy** *a* ∈ **𝒜** = {raw, cot, react, multiagent} **before** execution.

Executing *a* yields ŷₐ(*q*), exact-match *r*ₐ(*q*) ∈ {0,1}, and cost *c*ₐ(*q*) ≥ 0 (USD from token usage). A **routing policy** returns ŷ(*q*) = ŷ_{â(*q*)}(*q*); exactly one strategy runs at **deployment**.

Features φ(*q*) are defined in §3.

### 2.2 Utility, Oracle, and Regret

Utility (λ = 25):

\[
U_a(q) = r_a(q) - \lambda\, c_a(q), \qquad a^\star(q) \in \arg\max_a U_a(q).
\]

**C**(*q*) = {*a* ∈ 𝒜 : *r*ₐ(*q*) = 1}. Regret:

\[
\mathrm{Regret}(q) = U^\star(q) - U_{\hat{a}(q)}(q).
\]

Agent regret (execution only) and total regret (+ QCE at deployment). Macro-F1 over *a*⋆ diagnostic.

### 2.3 Routing Policy and Supervision

Router: *p*θ(*a* | φ(*q*)), â(*q*) = argmax *p*θ. **Cost-soft** *p*⋆ on **C**(*q*):

\[
p^\star(a \mid q) \propto \exp(-\tau\, c_a(q)), \quad a \in \mathcal{C}(q).
\]

Train by minimizing KL(*p*⋆ ∥ *p*θ). Problem: learn *p*θ, route before execution, evaluate regret (routing benchmark) and EM/cost (transfer).

---

## 3 Methodology

Pre-inference routing faces three challenges: estimate difficulty before execution; represent procedural and semantic signals; supervise when multiple correct strategies differ in cost. **AAR** addresses them via QCE, complexity–semantic features φ(*q*), and cost-soft targets (Figure 2).

**Figure 2.** Pipeline: query → planner → procedure DAG *G*_q → QCE → **z**(*q*); parallel semantic ψ(*q*) → φ(*q*) → router → argmax strategy → single execution.

### 3.1 Query Complexity Estimation

**Motivation.** Similar wording can imply different procedures (retrieval-heavy ReAct vs. reasoning-only CoT). Embeddings do not observe anticipated structure.

**Design.** LLM planner → procedure DAG *G*_q = (*V*, *E*) with subtask types retrieve, reason, compute, verify, select.

**Formalization.** QCE maps (*G*_q, plan) to **z**(*q*) ∈ ℝ⁵:

| Dimension | Description |
|-----------|-------------|
| *z*₁ structural | Depth and branching |
| *z*₂ reasoning | Multi-step inference |
| *z*₃ evidence | Retrieval demand |
| *z*₄ tool | External tool dependence |
| *z*₅ coordination | Subtask interaction |

**Benefit.** Consequently, heterogeneous plans become a compact pre-inference signal.

### 3.2 Complexity–Semantic Representation

\[
\phi(q) = [\mathbf{z}(q);\, \psi(q)]
\]

ψ(*q*): PCA-16 query embedding. Total dim. = 21 (Table 2). No GNN at inference.

| Feature group | Dimensions |
|---------------|----------:|
| **z**(*q*) | 5 |
| ψ(*q*) | 16 |
| **Total** | **21** |

### 3.3 Cost-Soft Supervision

**Motivation.** Hard labels discard cheaper correct strategies in **C**(*q*).

For |**C**(*q*)| ≥ 2:

\[
p^\star(a \mid q) = \frac{\exp(-\tau\, c_a(q))}{\sum_{a' \in \mathcal{C}(q)} \exp(-\tau\, c_{a'}(q))}, \quad p^\star(a)=0 \text{ if } a \notin \mathcal{C}(q).
\]

τ = 100 (label temperature); λ = 25 (utility)—distinct parameters.

| Scenario | **C**(*q*) | Target |
|----------|-----------|--------|
| None | ∅ | Excluded |
| One | {react} | One-hot |
| Multiple | {cot, react} | Cost-soft on **C** only |

### 3.4 Router Learning

\[
\mathcal{L}(\theta) = \mathbb{E}_q\big[\mathrm{KL}(p^\star(\cdot \mid q)\,\|\,p_\theta(\cdot \mid q))\big]
\]

HGBM and logistic regression as tabular backends; claim is supervision + features.

### 3.5 Inference

â(*q*) = argmaxₐ *p*θ(*a* | *q*). Only â runs at deployment.

---

## 4 Experiments

**We next ask:** (Q1) Does cost-soft supervision help? (Q2) Do procedure features help? (Q3) Does training transfer to live benchmarks?

### 4.1 Setup

- **Routing benchmark:** train 808 / val 100 / test 100; all four strategies executed offline per query; regret primary metric.
- **Transfer:** GAIA, HotpotQA, MuSiQue, MMLU-Pro, MATH (965 queries); executed EM and mUSD/query.
- **Baselines:** fixed strategies; hard / utility-softmax / cost-soft routers; heuristic features; oracle best-agent.
- **Representations:** embedding-only, heuristic, complexity-only, complexity–semantic (Appendix).

### 4.2 Summary of Findings

**Main:** cost-soft supervision → largest regret drop (0.361→0.192 agent); complexity helps when **combined** with embeddings (0.198 total regret); complexity-only worse than embedding-only (0.273 vs. 0.242).

**Transfer:** improvements do not fully translate (48.1% vs. 50.3% pooled EM).

### 4.3 Representation Analysis

**Research question:** Does planner-derived complexity help when fused with embeddings, relative to either alone? (All rows: HGBM, cost-soft *p*⋆, held-out test.)

| Representation | Total regret ↓ | Agent regret ↓ | EM ↑ |
|----------------|---------------:|---------------:|-----:|
| Embedding-only (ψ) | 0.242 | 0.242 | 77% |
| Heuristic | 0.240 | 0.240 | 77% |
| Complexity-only (z) | 0.273 | 0.267 | 74% |
| **Complexity–semantic (z+ψ)** | **0.198** | **0.192** | **82%** |

| Representation | Macro-F1 ↑ |
|----------------|----------:|
| Embedding-only | 0.261 |
| Heuristic | 0.255 |
| Complexity-only (z) | 0.290 |
| **Complexity–semantic (z+ψ, proposed)** | **0.323** |

**Takeaway:** z alone underperforms ψ alone on both regret metrics (total 0.273 / agent 0.267 vs.\ 0.242 / 0.242); fused φ(q)=[z;ψ] is best (total 0.198 / agent 0.192, 82% EM)—complexity is complementary, not sufficient by itself.

**QCE dimension ablation (Table 7 / Fig. 6):** single dim + ψ — Full 0.198; Structural 0.208; Reasoning 0.210; Evidence 0.228; Tool 0.247; Coordination 0.249. No single dim matches full fusion; structural/reasoning strongest alone; tool/coordination need combination. LOO: drop reasoning ≈ full (0.199); drop structural hurts most (0.228). See appendix.

### 4.4 Supervision Analysis

**Research question:** Does cost-soft beat hard labels and utility-softmax?

| Policy target | Classifier | Total regret ↓ | Agent regret ↓ | EM ↑ |
|---------------|------------|---------------:|---------------:|-----:|
| Hard | HGBM | 0.366 | 0.361 | 63% |
| Hard | Logistic | 0.251 | 0.245 | 76% |
| Soft (*p*⋆) | Logistic | 0.205 | 0.199 | 81% |
| **Soft (*p*⋆)** | **HGBM** | **0.198** | **0.192** | **82%** |

| Target | Total regret ↓ | Agent regret ↓ | EM ↑ |
|--------|---------------:|---------------:|-----:|
| Hard labels (HGBM) | 0.366 | 0.361 | 63% |
| Utility-softmax | 0.243 | 0.237 | 77% |
| **Cost-soft *p*⋆** | **0.198** | **0.192** | **82%** |

**Takeaway:** Supervision dominates classifier choice; cost-soft restricts mass to **C**(*q*).

### 4.5 Benchmark Transfer

| Workload | Proposed | Heur. | ReAct | CoT |
|----------|---------:|------:|------:|----:|
| GAIA | 18.8 | 24.9 | **26.1** | 6.1 |
| HotpotQA | 58.5 | **64.0** | **69.0** | 36.5 |
| MuSiQue | **36.0** | 35.0 | **36.5** | 2.5 |
| MMLU-Pro | 55.0 | **56.0** | **59.0** | 53.0 |
| MATH | 67.0 | 67.0 | 63.0 | **69.5** |
| **Pooled** | 48.1 | **50.3** | — | — |

**These results suggest** routing-benchmark gains do not uniformly transfer; validate per workload.

### 4.6 Discussion

**What works:** cost-soft supervision (0.361→0.192 agent regret); best config combines cost-soft + complexity–semantic (0.198 total).

**What partially works:** complexity complementary when fused; insufficient alone (0.273 vs. 0.242 embedding-only).

**What remains difficult:** transfer does not fully translate (48.1% vs. 50.3% pooled EM).

---

## 5 Related Work

- **Cost-aware routing:** FrugalGPT, RouteLLM, HybridLLM, RouterDC — model/API tiers; we route agent *strategies*.
- **Fusion:** LLM-Blender — post-hoc rank/fuse; we pre-inference route with cost-soft over **C**(*q*).
- **Graph routers:** GraphRouter, AgentRouter — affiliation graphs; we use procedure DAG complexity without GNN inference.
- **Difficulty / workflows:** DAAO, STRIDE — complementary pre-inference signals.

Full mapping: Appendix H.

---

## 6 Conclusion

**Problem:** static strategy selection misaligns cost and procedure demands. **Method:** AAR with cost-soft supervision and complexity–semantic features. **Finding:** cost-soft contributes most to regret reduction; complexity helps when combined, not alone. **Future:** transfer, cheaper complexity features, richer strategy pools, Pareto analysis.

---

## 7 Limitations

1. Strongest gains under oracle supervision; transfer trails heuristic EM on 4/5 families.
2. Planner cost and quality at inference.
3. Fixed four-strategy pool, shared backbone.
4. English-centric benchmarks.
5. Literature baselines not drop-in reimplemented (task mismatch).

---

## Appendix A — Benchmark Workloads

| Workload | Queries |
|----------|--------:|
| GAIA | 165 |
| HotpotQA | 200 |
| MuSiQue | 200 |
| MMLU | 200 |
| MATH | 200 |
| **Pooled** | **965** |

Pooled utility (λ=25): proposed **0.440** vs. heuristic **0.458**; bootstrap 95% CI [−0.0006, +0.035], *p* ≈ 0.059.

## Appendix B — Routing Benchmark

Stratified split 808/100/100 from HotpotQA, MuSiQue, MATH. Oracle log: four strategies × correctness × cost per query.

## Appendix C — Features

| Representation | ID | Dim. | Planner? |
|----------------|-----|-----:|:--------:|
| Embedding-only | emb_only | 16 | No |
| Heuristic | heur_emb | 50 | No |
| **Complexity–semantic** | **cvec5_emb** | **21** | **Yes** |

## Appendix D — Hyperparameters

| Parameter | Value |
|-----------|------:|
| λ (utility) | 25 |
| τ (cost-soft) | 100 |
| PCA | 16 |
| Primary model | hgbm_cvec5_emb_soft_kl.joblib |

Freeze manifest: `results/experiments/freeze/LATEST_FREEZE.json`.

## Appendix E — Additional Analyses

Calibration, cost–accuracy frontier, routing distributions, PCA sweep (8/16/32/64).

## Appendix F — Fixed-Strategy EM

(Same as Figure 1 table.)

## Appendix G — Selective QCE

Optional gate skips planner when embedding-only confidence is high.

## Appendix H — Related Work (Extended)

| Baseline | Maps to |
|----------|---------|
| RouteLLM | Utility regret |
| LLM-Blender | Cost-soft *p*⋆ |
| GraphRouter | Procedure **z**(*q*) + ψ |

---

## References

See [`QCE_Router_Paper.bib`](QCE_Router_Paper.bib). Compile with `pdflatex` + `bibtex`.

---

*Numbers aligned with frozen evaluation (`results/experiments/freeze/LATEST_FREEZE.json`, 2026-06-02).*
