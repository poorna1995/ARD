"""Tests for dataset-gated hop profiles and planner repair."""

from __future__ import annotations

import unittest

from agent.dataset_profile import (
    RUNTIME_HOP_KEYS,
    agent_kwargs_from_row,
    apply_multiagent_profile,
    apply_react_profile,
    extract_hop_metadata,
    is_open_wiki_qa,
    observation_text_from_tools_results,
    promote_hop_metadata,
    react_max_steps_for_dataset,
    validate_planner_subtasks,
)


def _benchmark_hop_kwargs(metadata: dict | None) -> dict:
    """Mirror run_training_benchmark._strategy_kwargs hop promotion."""
    hop_kw = agent_kwargs_from_row({"metadata": metadata or {}})
    return {k: hop_kw[k] for k in RUNTIME_HOP_KEYS if k in hop_kw}


class DatasetProfileTests(unittest.TestCase):
    def test_musique_react_defaults(self) -> None:
        p = apply_react_profile("musique", {}, n_hops=4)
        self.assertGreaterEqual(p["max_steps"], 16)
        self.assertEqual(react_max_steps_for_dataset("musique", 4), 16)
        self.assertEqual(react_max_steps_for_dataset("hotpot", 2), 10)
        self.assertIsNone(react_max_steps_for_dataset("math", 4))

    def test_musique_multiagent_defaults(self) -> None:
        p = apply_multiagent_profile("musique", {}, n_hops=4)
        self.assertGreaterEqual(p["tool_max_steps"], 10)
        self.assertEqual(p["max_subtasks"], 4)
        self.assertEqual(p["worker_result_forward_chars"], 300)

    def test_math_profile_unchanged(self) -> None:
        p = apply_react_profile("math", {}, n_hops=4)
        self.assertNotIn("max_steps", p)
        m = apply_multiagent_profile("gaia", {"max_workers": 4}, n_hops=4)
        self.assertEqual(m, {"max_workers": 4})

    def test_linear_chain_repair(self) -> None:
        subtasks = [
            {
                "id": "s1",
                "goal": "a",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
            {
                "id": "s2",
                "goal": "b",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
        ]
        out = validate_planner_subtasks(
            subtasks, dataset="musique", n_hops=4, hop_name="linear",
        )
        self.assertEqual(out[1]["depends_on"], ["s1"])

    def test_branching_skips_repair(self) -> None:
        subtasks = [
            {
                "id": "s2A",
                "goal": "a",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
            {
                "id": "s2B",
                "goal": "b",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
        ]
        validate_planner_subtasks(
            subtasks, dataset="musique", n_hops=3, hop_name="branching",
        )
        self.assertEqual(subtasks[0]["depends_on"], [])
        self.assertEqual(subtasks[1]["depends_on"], [])

    def test_parallel_skips_repair(self) -> None:
        subtasks = [
            {
                "id": "s1",
                "goal": "a",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
            {
                "id": "s2",
                "goal": "b",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
        ]
        validate_planner_subtasks(
            subtasks, dataset="musique", n_hops=2, hop_name="parallel",
        )
        self.assertEqual(subtasks[1]["depends_on"], [])

    def test_agent_kwargs_from_row(self) -> None:
        kw = agent_kwargs_from_row(
            {
                "metadata": {"n_hops": 3.0, "hop_name": "linear"},
            },
            expected_answer="x",
        )
        self.assertEqual(kw["n_hops"], 3)
        self.assertEqual(kw["hop_name"], "linear")
        self.assertEqual(kw["expected_answer"], "x")

    def test_benchmark_hop_kwargs_scenarios(self) -> None:
        musique_meta = {
            "n_hops": 4.0,
            "hop_name": "linear",
            "training_id": "train_1002",
        }
        hops = _benchmark_hop_kwargs(musique_meta)
        self.assertEqual(extract_hop_metadata(hops), (4, "linear"))
        profile = apply_multiagent_profile("musique", {}, n_hops=hops["n_hops"])
        self.assertEqual(profile["max_subtasks"], 4)

        hotpot = _benchmark_hop_kwargs({"n_hops": 2, "hop_name": "linear"})
        self.assertEqual(extract_hop_metadata(hotpot), (2, "linear"))
        self.assertEqual(react_max_steps_for_dataset("hotpot", hotpot["n_hops"]), 10)
        self.assertEqual(
            react_max_steps_for_dataset("musique", hops["n_hops"]),
            16,
        )

        math = _benchmark_hop_kwargs({"training_id": "train_1"})
        self.assertEqual(extract_hop_metadata(math), (None, None))
        self.assertNotIn("max_steps", apply_react_profile("math", {}, n_hops=None))

        branching = _benchmark_hop_kwargs(
            {"n_hops": 3, "hop_name": "branching"},
        )
        subtasks = [
            {
                "id": "s1",
                "goal": "a",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
            {
                "id": "s2",
                "goal": "b",
                "needs_tool": True,
                "tool_name": "wikipedia_search",
                "depends_on": [],
            },
        ]
        validate_planner_subtasks(
            subtasks,
            dataset="musique",
            n_hops=branching["n_hops"],
            hop_name=branching["hop_name"],
        )
        self.assertEqual(subtasks[1]["depends_on"], [])

        self.assertFalse(is_open_wiki_qa("math"))
        self.assertTrue(is_open_wiki_qa("musique"))

    def test_promote_hop_metadata_coerces(self) -> None:
        kw: dict[str, object] = {}
        promote_hop_metadata(kw, {"n_hops": 4.0, "hop_name": " Linear "})
        self.assertEqual(kw["n_hops"], 4)
        self.assertEqual(kw["hop_name"], "linear")

    def test_observation_content_extraction(self) -> None:
        text = observation_text_from_tools_results(
            [
                {
                    "action": "wikipedia_search",
                    "observation": '{"title":"T","content":"Fact here."}',
                },
            ],
        )
        self.assertIn("Fact here", text)


if __name__ == "__main__":
    unittest.main()
