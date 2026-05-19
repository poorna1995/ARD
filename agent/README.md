# Agents

Six agent strategies share the same **run entry point**, **config normalization**, and **`AgentResponse`** shape: `raw`, `cot`, `self_consistency`, `react`, `debate`, `multiagent`.

how should i have to meausre the query compelcity score for the given task dataset which help to router the suitable agent design worflow : Agents Let Agent A : a1,a2,a3,a4..an. i have choosed

the agent desgins / workflow which consistes raw llm, CoT,

Self consistency, ReAct, debate, plan and verify , which will effecitvely help reduce the costs.

find the best method to measure teh complexity of a query befoere passing to the agent design.

search in web, dont halluciante, and dont manipulate.

lists all the relevant methods to my research work

## Quick start

```python
from agent.registry import run_strategy

resp = run_strategy(
    "cot",                              # agent name
    query="What is the capital of France?",
    model="llama-3.3-70b-versatile",    # or gpt-4o, gpt-4o-mini, …
    dataset="hotpot",                   # gaia | mmlu_pro | math | hotpot | musique
    expected_answer="Paris",            # optional — sets is_correct
    temperature=0.1,
    max_tokens=1024,
    seed=42,
)
print(resp.predicted_answer, resp.confidence, resp.is_correct)
```

Smoke-test all six agents:

```bash
uv run python scripts/test_agents.py --query "What is the capital of France?" --dataset hotpot
```

---

## 1. Shared inputs

Every agent’s `run()` accepts the same **runtime arguments**.

| Field             | Type  | Required | Description                                                                                                                                                      |
| ----------------- | ----- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `query`           | `str` | yes      | Full task text from parquet / loader. Only `{query}` is substituted into user prompts. Options, file names, and context are embedded in this string when needed. |
| `model`           | `str` | yes      | LLM id. Groq: `llama-3.3-70b-versatile`, `deepseek-r1`. OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-o1`.                                                               |
| `dataset`         | `str` | yes      | Selects prompts in `prompts/prompts.py`. One of: `gaia`, `mmlu_pro`, `math`, `hotpot`, `musique`.                                                                |
| `expected_answer` | `str` | no       | Gold label for eval; sets `is_correct` on the response.                                                                                                          |
| `**kwargs`        | —     | no       | Common config + agent-specific params (below).                                                                                                                   |

**Environment (not passed per call):**

- `GROQ_API_KEY` — Groq models
- `OPENAI_API_KEY` — OpenAI models
- Tool keys as needed (`SERPER_API_KEY` / `SERP_API_KEY` for SerpAPI-backed `web_search`, etc.)

`.env` is loaded via `dotenv` in `agent/base.py` and `agent/tools/search.py`.

---

## 2. Shared configuration

Normalized by `agent/config.py` → `normalize_agent_config()`.

### Common (all agents)

| Parameter     | Default | Description                                                                              |
| ------------- | ------- | ---------------------------------------------------------------------------------------- |
| `temperature` | `0.1`   | Main LLM sampling temperature (debate / self-consistency may override internally).       |
| `max_tokens`  | `1024`  | Max completion tokens per call. Multi-agent uses role-specific caps unless overridden.   |
| `seed`        | `42`    | Reproducibility (`None` to disable). Self-consistency uses `seed + path_index` per path. |

### Self-consistency only

| Parameter            | Default  | Description                                                                                                  |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `num_paths`          | `5`      | Independent CoT samples to generate.                                                                         |
| `sample_temperature` | `0.7`    | Temperature for each path (min effective `0.5` if set &lt; 0.01).                                            |
| `vote_key_strategy`  | `"auto"` | Majority vote key: `canonical`, `literal_ci`, or `auto` (literal for `math` / formula / chess-like queries). |
| `log_sc_paths`       | `true`   | Log path details to stderr.                                                                                  |

### Multi-agent only

| Parameter                       | Default         | Description                                           |
| ------------------------------- | --------------- | ----------------------------------------------------- |
| `max_workers`                   | `4`             | Max subtasks executed per plan.                       |
| `max_subtasks`                  | `5`             | Max subtasks returned by planner.                     |
| `enable_tools_when_needed`      | `true`          | Run ReAct tool worker when planner sets `needs_tool`. |
| `tool_max_steps`                | `5`             | Max ReAct steps per tool worker.                      |
| `planner_model`                 | `gpt-4o-mini`   | Planner LLM.                                          |
| `worker_model`                  | `gpt-4o-mini`   | LLM-only worker LLM.                                  |
| `judge_model`                   | same as `model` | Judge LLM.                                            |
| `worker_retry_threshold`        | `0.4`           | Low-confidence worker retry threshold.                |
| `worker_result_forward_chars`   | `200`           | Cap on worker result text forwarded in memory.        |
| `worker_evidence_forward_chars` | `80`            | Cap on evidence snippets in memory.                   |

**Aliases:** `max_tool_steps` → `tool_max_steps`, `cheap_model` → `worker_model`, `verifier_model` → `judge_model`.

**Defined in config but not wired in `multiagent.py` yet:** `max_retries`, `max_replan`.

### ReAct only (constructor / kwargs)

| Parameter   | Default | Description                               |
| ----------- | ------- | ----------------------------------------- |
| `max_steps` | `12`    | Max Thought → Action → Observation loops. |

### Debate (fixed in code, not kwargs)

| Setting                | Value       | Notes                                                              |
| ---------------------- | ----------- | ------------------------------------------------------------------ |
| Debater temperature    | dataset map | `gaia`/`hotpot`/`musique`: `0.7`; `mmlu_pro`: `0.5`; `math`: `0.3` |
| Synthesis temperature  | `0.0`       | When debaters disagree                                             |
| Debater `max_tokens`   | `1200`      | Per debater call                                                   |
| Synthesis `max_tokens` | `800`       | Judge synthesis call                                               |

---

## 3. LLM output schema (prompt contract)

Most agents ask the model for one JSON object:

```json
{ "answer": "<value>", "confidence": 0.0, "complexity": 0.0 }
```

Parsed by `evaluator.eval.parse_agent_output` / `normalise_answer`.  
`AgentResponse.predicted_answer` is the **bare** string (not the full JSON). Gold labels stay in parquet column `answer` → `expected_answer`.

**ReAct / GAIA-style tool finish:** `Action: finish[{"answer":"...","confidence":0.9,"complexity":0.3}]` — same three fields, validated on every dataset.

**Multi-agent judge (internal)** also returns `consensus_score`, `rationale`, `conflict_resolution`; only `predicted_answer`, `confidence`, `complexity`, and `consensus_score` are promoted to `AgentResponse`.

---

## 4. Shared output: `AgentResponse`

Returned by every agent (`agent/base.py`).

### Core (always set)

| Field              | Type          | Description                                              |
| ------------------ | ------------- | -------------------------------------------------------- |
| `query`            | `str`         | Input question                                           |
| `predicted_answer` | `str`         | Parsed model output (bare value, from JSON key `answer`) |
| `model`            | `str`         | Model used                                               |
| `agent`            | `str`         | Strategy name (`raw`, `cot`, …)                          |
| `agent_id`         | `str`         | Stable id (`raw_001`, …)                                 |
| `dataset`          | `str`         | Dataset key                                              |
| `is_failed`        | `bool`        | True on crash / empty answer where applicable            |
| `error`            | `str \| None` | Error message if failed                                  |

### Latency & cost

| Field               | Default | Description                               |
| ------------------- | ------- | ----------------------------------------- |
| `latency_total`     | `0.0`   | Wall time (seconds)                       |
| `latency_llm`       | `0.0`   | Time in LLM calls                         |
| `latency_tools`     | `0.0`   | Time in tool calls (ReAct, multi-agent)   |
| `prompt_tokens`     | `0`     | Sum of prompt tokens                      |
| `completion_tokens` | `0`     | Sum of completion tokens                  |
| `total_tokens`      | `0`     | prompt + completion                       |
| `cost_usd`          | `0.0`   | Estimated from `COST_PER_1M` in `base.py` |

### Evaluation

| Field             | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| `expected_answer` | Gold label if passed in                                                              |
| `is_correct`      | `True`/`False`/`None` — `math` uses sympy equivalence; others canonical string match |
| `confidence`      | From model JSON when present                                                         |
| `complexity`      | From model JSON when present                                                         |

### Reasoning & tools (populated per agent)

| Field              | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `reasoning_steps`  | CoT lines, debate trace, subtask goals, or winning SC path steps |
| `num_llm_calls`    | Number of LLM API calls                                          |
| `num_steps`        | Agent turns (1 for single-shot raw/cot; ReAct tool loops; etc.)  |
| `max_steps`        | Configured cap (ReAct / multi-agent)                             |
| `steps_taken`      | Turns or paths actually used (matches agent semantics)           |
| `is_stopped_early` | ReAct hit `max_steps` without valid `finish`                     |
| `tools_available`  | Tool names registered for this run                               |
| `tools_called`     | Tools invoked                                                    |
| `tools_results`    | Per-step tool I/O (ReAct steps; multi-agent tool workers)        |
| `num_tool_calls`   | Count of non-finish tool calls                                   |

### Multi-agent / debate / self-consistency

| Field                 | Description                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `orchestration_type`  | e.g. `none`, `multi_agent_debate`, `self_consistency`, `planner_workers_judge_working_memory_v6_optimised` |
| `sub_agents_run`      | Roles executed: `planner`, `worker_s1`, `judge`, `agent_a`, `sc_cot_path_0`, …                             |
| `sub_agent_responses` | Raw structured outputs per sub-role                                                                        |
| `selected_agent`      | e.g. `agreement`, `synthesis`, `majority_vote`, `judge`                                                    |
| `consensus_score`     | Agreement strength (debate / SC / multi-agent judge)                                                       |

---

## 5. Per-agent reference

### 5.1 `raw` (`raw_001`)

**What it is:** Single-shot direct answer — no explicit reasoning, no tools.

**Config:** Common only.

**Process (mid-steps):**

1. Build messages: `SYSTEM_PROMPT[dataset]["raw"]` + `USER_PROMPT[dataset]["raw"]` with `{query}`.
2. One `chat.completions.create`.
3. `parse_llm_output` on raw text → `predicted_answer`, `confidence`, `complexity`.

**Typical LLM calls:** 1.

**Outputs highlights:** `reasoning_steps=[]`, `orchestration_type="none"`, `num_steps=1`.

---

### 5.2 `cot` (`cot_002`)

**What it is:** Chain-of-thought — model reasons in prose, then JSON answer.

**Config:** Common only.

**Process:**

1. Same message pattern with `cot` prompts.
2. One LLM call.
3. Split response: lines before `{` → `reasoning_steps`; trailing JSON → parsed fields.

**Typical LLM calls:** 1.

**Outputs highlights:** `reasoning_steps` = CoT lines; `num_steps` / `steps_taken` = 1 (one LLM call).

---

### 5.3 `self_consistency` (`self_consistency_003`)

**What it is:** Multiple independent CoT samples + majority vote (Wang et al., 2023).

**Config:** Common + `num_paths`, `sample_temperature`, `vote_key_strategy`, `log_sc_paths`.

**Process:**

1. For `path` in `0 .. num_paths-1`:
   - LLM call at `sample_temperature` (seed offset per path).
   - Parse JSON answer → `vote_key` (canonical or literal).
   - Store path row in `sub_agent_responses`.
2. Majority vote on `vote_key` → winner `answer`.
3. `consensus_score` = winner count / usable paths.
4. `confidence` = average confidence of paths that voted for winner.

**Typical LLM calls:** `num_paths` (default 5).

**Outputs highlights:**

| Field                 | Value                                                   |
| --------------------- | ------------------------------------------------------- |
| `complexity`          | Usually `None` on final response                        |
| `orchestration_type`  | `"self_consistency"`                                    |
| `sub_agents_run`      | `sc_cot_path_0`, …                                      |
| `selected_agent`      | `"majority_vote"`                                       |
| `sub_agent_responses` | Per-path dicts (`parsed_answer`, `vote_key`, tokens, …) |

---

### 5.4 `debate` (`debate_006`)

**What it is:** Two independent debaters; synthesis only if answers disagree after canonicalisation.

**Config:** Common + fixed debate temperatures (see table above).

**Process:**

1. **Round 1 — parallel debaters** (same user prompt, dataset debater temperature):
   - Agent A → parse JSON.
   - Agent B → parse JSON.
   - Append to `reasoning_steps` / `sub_agent_responses`.
2. **If canonical answers match:** return A’s answer, `consensus_score=1.0`, `selected_agent="agreement"` (2 LLM calls).
3. **If disagree — synthesis** (temp `0.0`, `DEBATE_SYNTHESIS_*` prompts):
   - Third LLM call merges A/B traces.
   - `selected_agent="synthesis"`, `consensus_score=0.6`.

**Typical LLM calls:** 2 (agree) or 3 (disagree).

**Outputs highlights:** `orchestration_type="multi_agent_debate"`; trace in `reasoning_steps`.

---

### 5.5 `react` (`react_004`)

**What it is:** ReAct loop — Thought → Action → Observation until `finish[JSON]`.

**Config:** Common + `max_steps` (default 12).

**Tools (GAIA-style datasets):** `web_search`, `web_fetch`, `wikipedia_search`, `arxiv_search`, `github_search`, `pdb_parse`, `read_file`, `math_tool`, `finish`.

**Hotpot / musique / math / mmlu_pro:** prompts use internal `reason[]` or `calculate[]` actions (no external tools in system prompt).

**Process (each step):**

1. LLM generates `Thought:` + `Action: tool[input]`.
2. If action is `finish`: validate JSON schema (`answer`, `confidence`, `complexity`); retry on invalid payload.
3. Else: run tool → `Observation:` appended as user message.
4. Repeat until valid finish or `max_steps` (then may fallback-parse last output).

**Typical LLM calls:** 1–`max_steps` (+ tool latency in `latency_tools`).

**Outputs highlights:**

| Field                            | Notes                                  |
| -------------------------------- | -------------------------------------- |
| `tools_called` / `tools_results` | Per-step action log                    |
| `is_stopped_early`               | True if max steps without valid finish |
| `confidence` / `complexity`      | From finish JSON                       |

---

### 5.6 `multiagent` (`multiagent_007`)

**What it is:** Planner → Workers → Judge with shared working memory.

**Config:** Common + multi-agent table above.

**Process:**

1. **Planner** (`planner_model`): JSON subtask list (`id`, `goal`, `focus`, `needs_tool`, `depends_on`, …).
2. **Workers** (up to `max_workers` subtasks):
   - **Reused:** skip if dependency result already in memory.
   - **Tool worker:** nested `ReactAgent` when `needs_tool` and `enable_tools_when_needed`.
   - **LLM worker:** `worker_model`, compact JSON result.
   - Update working memory (`known_facts`, `tool_summaries`, …).
3. **Judge** (`judge_model`): reads memory + subtasks → bare `answer` + `confidence`, `complexity`, `consensus_score`.

**Typical LLM calls:** 1 planner + N workers + 1 judge (+ extra calls inside tool workers).

**Outputs highlights:**

| Field                 | Notes                                |
| --------------------- | ------------------------------------ |
| `predicted_answer`    | Judge’s bare answer string           |
| `reasoning_steps`     | List of subtask `goal` strings       |
| `sub_agent_responses` | Planner / worker / judge payloads    |
| `tools_*`             | Aggregated from tool-enabled workers |

---

## 6. Registry summary

| Agent              | `agent_id`             | Tools                   | Typical LLM calls |
| ------------------ | ---------------------- | ----------------------- | ----------------- |
| `raw`              | `raw_001`              | No                      | 1                 |
| `cot`              | `cot_002`              | No                      | 1                 |
| `self_consistency` | `self_consistency_003` | No                      | `num_paths`       |
| `react`            | `react_004`            | Yes (dataset-dependent) | 1–12              |
| `debate`           | `debate_006`           | No                      | 2–3               |
| `multiagent`       | `multiagent_007`       | Optional per subtask    | 3+                |

---

## 7. Prompts & datasets

- Prompts: `prompts/prompts.py` — `SYSTEM_PROMPT[dataset][agent]`, `USER_PROMPT[dataset][agent]`.
- Multi-agent role prompts: `build_multiagent_prompts(dataset, tools_csv, tool_enum)`.
- Validated datasets: `gaia`, `mmlu_pro`, `math`, `hotpot`, `musique`.

---

## 8. Related files

| File                     | Role                                      |
| ------------------------ | ----------------------------------------- |
| `agent/registry.py`      | `run_strategy()`, `STRATEGY_REGISTRY`     |
| `agent/base.py`          | `BaseAgent`, `AgentResponse`, `_call_llm` |
| `agent/config.py`        | Config normalization                      |
| `evaluator/eval.py`      | `parse_agent_output`, `is_correct`        |
| `scripts/test_agents.py` | Smoke-test all six agents                 |
| `scripts/test_tools.py`  | Smoke-test ReAct tools                    |
