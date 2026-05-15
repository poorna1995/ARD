from __future__ import annotations

import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agent.base import BaseAgent, AgentResponse
from evaluator.eval import canonicalise_answer, is_correct, parse_agent_output
from prompts.prompts import SYSTEM_PROMPT, USER_PROMPT

# Logic / math symbols in the *question* → use literal vote keys (canonical strips `a`/`an`
# as articles and nukes operators, collapsing different formulas to the same key).
# Chess algebraic notation (e.g. Qe1#) loses `#`/`+` under canonical — literal_ci keeps them.
_FORMULA_HINT_RE = re.compile(
    r"[¬→↔∧∨∀∃⊢≡≠≤≥∈∉∪∩⇒⇔∄⊂⊃⊆⊇∅∑∏√∞±×÷]"
)
_CHESS_CONTEXT_RE = re.compile(
    r"(?i)\b(chess|checkmate|stalemate|en passant|castl(?:e|ing)|"
    r"white to move|black to move|best move|mate in)\b"
)
_CHESS_SAN_RE = re.compile(
    r"(?i)\b[KQRBN][a-h][1-8](?:[+#]|=[QRBN])?\b|"
    r"\b[a-h]x[a-h][1-8](?:[+#])?\b|"
    r"\b[a-h][1-8](?:[+#])\b|"
    r"\bO-O(?:-O)?\b|\b0-0(?:-0)?\b"
)


def _guess_formula_context(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return bool(_FORMULA_HINT_RE.search(q))


def _guess_chess_like_context(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if _CHESS_CONTEXT_RE.search(q):
        return True
    if _CHESS_SAN_RE.search(q):
        return True
    return False


def _auto_use_literal_vote_keys(query: str, dataset: str) -> bool:
    if dataset == "math_hard":
        return True
    return _guess_formula_context(query) or _guess_chess_like_context(query)


def _paths_suggest_chess_san(paths: list[dict[str, Any]]) -> bool:
    """Detect mate/check markers stripped by canonical (e.g. Qe1# → qe1)."""
    for p in paths:
        a = str(p.get("parsed_answer") or "").strip()
        if not a:
            continue
        if "#" in a and re.search(r"(?i)[KQRBN][a-h][1-8]", a):
            return True
        if re.search(r"(?i)[KQRBN][a-h][1-8]\+", a):
            return True
    return False


def _literal_ci_vote_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    return re.sub(r"\s+", " ", t)


def _vote_key_for_path(answer: str, strategy: str) -> str:
    s = (strategy or "canonical").strip().lower()
    if s == "literal_ci":
        return _literal_ci_vote_key(answer)
    return canonicalise_answer(str(answer or ""))


def _is_placeholder_parsed_answer(ans: str) -> bool:
    """Prompt-leak / refusal strings must not receive votes or win."""
    t = str(ans or "").strip().lower()
    if not t:
        return True
    if t in {"<value>", "<short value>", "unknown", "n/a", "na", "none", "null", "..."}:
        return True
    if len(t) >= 2 and t.startswith("<") and t.endswith(">"):
        return True
    return False


def _extract_reasoning(raw_answer: str) -> tuple[list[str], str]:
    """Split CoT lines from trailing JSON (same idea as ``agent.cot.CotAgent``)."""
    steps: list[str] = []
    lines = raw_answer.strip().split("\n")
    for line in lines:
        if line.strip().startswith("{") and "answer" in line:
            break
        if line.strip():
            steps.append(line.strip())
    match = re.search(r'\{"answer":\s*"([^"]+)"\}', raw_answer)
    final = match.group(0) if match else raw_answer
    return steps, final


def _majority_from_paths(
    paths: list[dict[str, Any]],
) -> tuple[str, float, str | None, dict[str, Any]]:
    usable = [p for p in paths if p.get("vote_key")]
    if not usable:
        return "", 0.0, None, {"counts": {}, "n_usable": 0}

    keys_all: list[str] = []
    first_idx_all: dict[str, int] = {}
    display_for_key: dict[str, str] = {}
    keys_votes: list[str] = []
    first_idx_vote: dict[str, int] = {}

    for p in sorted(usable, key=lambda x: int(x.get("path", 0))):
        k = str(p["vote_key"])
        if not k:
            continue
        pa = str(p.get("parsed_answer", ""))
        keys_all.append(k)
        if k not in first_idx_all:
            first_idx_all[k] = int(p.get("path", 0))
            display_for_key[k] = pa
        if not _is_placeholder_parsed_answer(pa):
            keys_votes.append(k)
            if k not in first_idx_vote:
                first_idx_vote[k] = int(p.get("path", 0))

    # Do not count placeholder answers toward majority; if all paths are placeholders, fall back.
    use_votes = keys_votes if keys_votes else keys_all
    first_idx = first_idx_vote if keys_votes else first_idx_all

    counts = Counter(use_votes)
    counts_all = Counter(keys_all)
    n_cast = len(use_votes)
    n_total = len(keys_all)

    max_votes = max(counts.values())
    candidates = [k for k, c in counts.items() if c == max_votes]
    winner_key = sorted(candidates, key=lambda k: first_idx[k])[0]

    win_display = display_for_key.get(winner_key, "")
    runner_up_used = False
    if _is_placeholder_parsed_answer(win_display):
        ranked = sorted(
            [
                (k, c)
                for k, c in counts_all.items()
                if not _is_placeholder_parsed_answer(display_for_key.get(k, ""))
            ],
            key=lambda kv: (-kv[1], first_idx_all.get(kv[0], 10**9)),
        )
        if ranked:
            winner_key = ranked[0][0]
            counts = counts_all
            n_cast = n_total
            first_idx = first_idx_all
            runner_up_used = True
        else:
            return "", 0.0, None, {
                "counts": dict(counts),
                "counts_all": dict(counts_all),
                "n_usable": 0,
                "n_paths_total": n_total,
                "excluded_placeholder_paths_from_vote": bool(keys_votes),
                "winner_votes": 0,
                "all_answers_placeholder": True,
            }

    consensus = float(counts[winner_key] / n_cast) if n_cast else 0.0
    top = max(counts.values())
    tied_final = [k for k, cv in counts.items() if cv == top]
    return display_for_key.get(winner_key, ""), consensus, winner_key, {
        "counts": dict(counts),
        "counts_all": dict(counts_all),
        "n_usable": n_cast,
        "n_paths_total": n_total,
        "excluded_placeholder_paths_from_vote": bool(keys_votes),
        "winner_votes": counts[winner_key],
        "tied_at_max": len(tied_final) > 1,
        "tied_candidates": tied_final if len(tied_final) > 1 else [],
        "runner_up_non_placeholder": runner_up_used,
    }


def _sc_log(msg: str) -> None:
    """Self-consistency trace lines (stderr so JSON/tqdm stdout stays clean)."""
    print(msg, file=sys.stderr, flush=True)


class SelfConsistencyAgent(BaseAgent):

    def __init__(self, model: str, dataset: str, **kwargs: Any) -> None:
        cfg_result = self._normalize_config(
            model=model,
            dataset=dataset,
            kwargs=kwargs,
        )
        cfg = cfg_result.config
        params = cfg.agent_params

        self.dataset = cfg.dataset
        self.num_paths = max(1, int(params.get("num_paths", 5)))
        self.sample_temperature = float(params.get("sample_temperature", 0.7))
        if self.sample_temperature < 0.01:
            self.sample_temperature = 0.5
        self._sc_log_paths = bool(params.get("log_sc_paths", True))
        self._vote_key_strategy_request = str(
            params.get("vote_key_strategy", "auto")
        ).strip().lower()

        sc_system = SYSTEM_PROMPT[self.dataset]["self_consistency"]
        sc_user = USER_PROMPT[self.dataset]["self_consistency"]

        super().__init__(
            model=cfg.model,
            system_prompt=sc_system,
            user_prompt=sc_user,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    def _call_llm_sample(
        self,
        query: str,
        *,
        temperature: float,
        path_index: int,
    ) -> tuple[str, float, Any]:
        start = time.perf_counter()
        path_seed: int | None = None
        if self.seed is not None:
            path_seed = int(self.seed) + int(path_index)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(query=query)},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if path_seed is not None:
            create_kwargs["seed"] = path_seed

        client = self._get_client()
        try:
            response = client.chat.completions.create(**create_kwargs)
        except Exception:
            create_kwargs.pop("seed", None)
            response = client.chat.completions.create(**create_kwargs)

        content = response.choices[0].message.content or ""
        return content, time.perf_counter() - start, response

    def _one_path(self, query: str, path_index: int) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": path_index,
            "parsed_answer": "",
            "vote_key": "",
            "confidence": None,
            "complexity": None,
            "reasoning_steps": [],
            "latency_llm": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "error": None,
            "raw_head": "",
        }
        try:
            raw, latency, response = self._call_llm_sample(
                query,
                temperature=self.sample_temperature,
                path_index=path_index,
            )
            steps, final_blob = _extract_reasoning(raw)
            answer, conf, comp = parse_agent_output(final_blob)
            msg = response.choices[0].message
            reasoning_steps = getattr(msg, "reasoning_steps", None) or steps
            usage = self._get_usage(response)
            pt = int(usage["prompt_tokens"])
            ct = int(usage["completion_tokens"])
            out.update(
                {
                    "parsed_answer": answer,
                    "vote_key": _vote_key_for_path(
                        answer,
                        getattr(self, "_vote_strategy_effective", "canonical"),
                    ),
                    "confidence": conf,
                    "complexity": comp,
                    "reasoning_steps": list(reasoning_steps),
                    "latency_llm": latency,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": int(usage["total_tokens"]),
                    "cost_usd": self._compute_cost(pt, ct),
                    "raw_head": (raw or "")[:800],
                }
            )
        except Exception as exc:  # noqa: BLE001 — record per-path failures for traces
            out["error"] = str(exc)
        return out

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        expected = kwargs.get("expected_answer")
        log_paths = bool(kwargs.get("log_sc_paths", self._sc_log_paths))

        vk_kw = kwargs.get("vote_key_strategy")
        raw_s = str(
            vk_kw if vk_kw is not None else self._vote_key_strategy_request
        ).strip().lower()
        if raw_s in ("auto", ""):
            eff = "literal_ci" if _auto_use_literal_vote_keys(query, self.dataset) else "canonical"
        elif raw_s in ("canonical", "literal_ci"):
            eff = raw_s
        else:
            eff = "canonical"
        self._vote_strategy_effective = eff

        t_wall0 = time.perf_counter()
        path_rows: list[dict[str, Any]] = []
        max_workers = min(self.num_paths, 8)

        if log_paths:
            _sc_log(
                f"[self_consistency] sampling {self.num_paths} paths "
                f"(T={self.sample_temperature:g}) …"
            )
            _sc_log(
                f"[self_consistency] vote_key_strategy={eff!r} "
                f"(vote_key_strategy param={raw_s!r})"
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._one_path, query, i): i
                for i in range(self.num_paths)
            }
            for fut in as_completed(futures):
                row = fut.result()
                path_rows.append(row)
                if log_paths:
                    err = row.get("error")
                    extra = f" | ERROR: {err}" if err else ""
                    vk = row.get("vote_key") or "(empty)"
                    pa = row.get("parsed_answer") if row.get("parsed_answer") else "(empty)"
                    _sc_log(
                        f"[self_consistency] path {row['path']} finished "
                        f"({row.get('latency_llm', 0):.2f}s) | "
                        f"parsed_answer={pa!r} | vote_key={vk!r}{extra}"
                    )

        path_rows.sort(key=lambda r: int(r["path"]))
        latency_llm = sum(float(r["latency_llm"]) for r in path_rows)

        if self._vote_strategy_effective == "canonical" and _paths_suggest_chess_san(path_rows):
            for r in path_rows:
                pa = str(r.get("parsed_answer") or "")
                if pa:
                    r["vote_key"] = _literal_ci_vote_key(pa)
            self._vote_strategy_effective = "literal_ci"
            if log_paths:
                _sc_log(
                    "[self_consistency] vote_key_strategy switched to 'literal_ci' "
                    "(chess-style SAN with #/+ detected; canonical would strip those)"
                )

        winner, consensus, wkey, vote_meta = _majority_from_paths(path_rows)
        vote_meta = {**vote_meta, "vote_key_strategy": self._vote_strategy_effective}

        if log_paths:
            _sc_log("[self_consistency] --- majority vote (by vote_key) ---")
            counts = vote_meta.get("counts") or {}
            if not counts:
                _sc_log("  (no usable votes — all paths empty or failed)")
            else:
                display_by_key: dict[str, str] = {}
                for r in path_rows:
                    k = str(r.get("vote_key") or "")
                    if k and k not in display_by_key:
                        display_by_key[k] = str(r.get("parsed_answer", ""))

                for vk, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    disp = display_by_key.get(vk, "")
                    _sc_log(
                        f"  {cnt} vote(s)  vote_key={vk!r}  "
                        f"display_answer={disp!r}"
                    )
                max_c = max(counts.values())
                tied_keys = [k for k, v in counts.items() if v == max_c]
                if len(tied_keys) > 1:
                    tie_suffix = (
                        " | tied at top count — earliest path index wins among tied keys"
                    )
                else:
                    tie_suffix = ""
                _sc_log(
                    f"[self_consistency] WINNER display_answer={winner!r}  "
                    f"consensus={consensus:.2f}  winner_vote_key={wkey!r}{tie_suffix}"
                )

        win_reasoning: list[str] = []
        confidences: list[float] = []
        if wkey is not None:
            for r in path_rows:
                if str(r.get("vote_key")) != str(wkey):
                    continue
                if not win_reasoning and r.get("reasoning_steps"):
                    win_reasoning = list(r["reasoning_steps"])
                c = r.get("confidence")
                if isinstance(c, (int, float)):
                    confidences.append(float(c))

        if not win_reasoning:
            for r in path_rows:
                if r.get("reasoning_steps"):
                    win_reasoning = list(r["reasoning_steps"])
                    break

        rows_out: list[dict[str, Any]] = [dict(r) for r in path_rows]
        if rows_out:
            rows_out[0]["vote_meta"] = vote_meta

        pt_tot = sum(int(r.get("prompt_tokens", 0)) for r in path_rows)
        ct_tot = sum(int(r.get("completion_tokens", 0)) for r in path_rows)
        tok_tot = sum(int(r.get("total_tokens", 0)) for r in path_rows)
        cost_tot = sum(float(r.get("cost_usd", 0.0)) for r in path_rows)
        errs = [r for r in path_rows if r.get("error")]

        avg_conf = (
            sum(confidences) / len(confidences) if confidences else None
        )

        response_obj = AgentResponse(
            query=query,
            answer=winner,
            model=self.model,
            agent="self_consistency",
            dataset=self.dataset,
            latency_total=time.perf_counter() - t_wall0,
            latency_llm=latency_llm,
            latency_tools=0.0,
            prompt_tokens=pt_tot,
            completion_tokens=ct_tot,
            total_tokens=tok_tot,
            cost_usd=cost_tot,
            reasoning_steps=win_reasoning,
            num_llm_calls=self.num_paths,
            num_steps=len(win_reasoning),
            tools_available=[],
            tools_called=[],
            tools_results=[],
            num_tool_calls=0,
            max_steps=self.num_paths,
            steps_taken=self.num_paths,
            is_stopped_early=False,
            orchestration_type="self_consistency",
            sub_agents_run=[f"sc_cot_path_{i}" for i in range(self.num_paths)],
            selected_agent="majority_vote",
            sub_agent_responses=rows_out,
            consensus_score=consensus,
            expected_answer=expected,
            is_correct=(
                is_correct(winner, str(expected)) if expected else None
            ),
            confidence=avg_conf,
            complexity=None,
            error=("; ".join(str(r["error"]) for r in errs) if errs else None),
            is_failed=bool(winner == "" and errs),
        )

        return self._finalize_response(response_obj)


def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
    agent = SelfConsistencyAgent(model=model, dataset=dataset, **kwargs)
    return agent.run(query, **kwargs)


if __name__ == "__main__":
    import pandas as pd

    ds = "gaia"
    m = "gpt-4o-mini"
    df = pd.read_parquet(
        f"datasets/golden/{ds}.parquet", columns=["query", "answer"]
    )
    ag = SelfConsistencyAgent(model=m, dataset=ds, num_paths=3, seed=0)
    for _, row in df.head(2).iterrows():
        print(
            ag.run(
                query=str(row["query"]),
                expected_answer=row.get("answer"),
            )
        )
