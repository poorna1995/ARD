"""
Task Decomposition Pipeline — DAG-based complexity analysis
Stages: Parse → Discover Relationships → Build DAG → Score
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import anthropic

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

class EdgeType(Enum):
    TEMPORAL   = "temporal"    # ordering constraint (must happen before/after)
    DATA_FLOW  = "data_flow"   # output of A is input to B
    SEMANTIC   = "semantic"    # SRL: A produces the entity B operates on


@dataclass
class SubTask:
    id: str                          # e.g. "s1"
    action_verb: str                 # e.g. "Search"
    target_noun: str                 # e.g. "arXiv paper"
    description: str                 # human-readable
    tool_needs: list[str] = field(default_factory=list)   # e.g. ["web_search", "pdf_reader"]
    in_degree: int = 0
    out_degree: int = 0
    # Complexity scores (0–10)
    R: float = 0.0   # reasoning depth
    T: float = 0.0   # tool / resource need
    S: float = 0.0   # state / memory need
    rho: float = 0.0 # risk / success probability (inverted: high rho = high risk)
    ASS: float = 0.0 # aggregate subtask score


@dataclass
class Edge:
    source: str       # subtask id
    target: str       # subtask id
    edge_type: EdgeType
    label: str = ""   # what data flows (for DATA_FLOW edges)


@dataclass
class DAG:
    subtasks: dict[str, SubTask] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=dict)
    query: str = ""
    aggregate_complexity: float = 0.0
    needs_agent: bool = False


# ─────────────────────────────────────────────
# Stage 1 — Parse query into subtasks via LLM
# ─────────────────────────────────────────────

PARSE_SYSTEM = """You are a task decomposition and workflow planning system.
your job is to break down a complex user query into atomic subtasks into a structured Directed Acyclic Graph (DAG) of subtasks  


Return ONLY valid JSON in this exact shape — no markdown, no explanation:
{
  "subtasks": [
    {
      "id": "s1",
      "action_verb": "Search",
      "target_noun": "arXiv database",
      "description": "Search arXiv for AI regulation papers from June 2022",
      "tool_needs": ["web_search", "arxiv_api"]
    }
  ]
}

Rules:
- Decompose until each subtask is truly atomic (cannot be split further)
- Assign sequential ids: s1, s2, s3 ...
- tool_needs must be specific: web_search, arxiv_api, pdf_reader, image_parser,
  code_executor, calculator, database_query, wikipedia, news_api, etc.
- If a subtask needs NO external tool (pure reasoning), use []
"""

def parse_query(client: anthropic.Anthropic, query: str) -> list[SubTask]:
    """Stage 1: LLM parses free-form query → structured subtask list."""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=PARSE_SYSTEM,
        messages=[{"role": "user", "content": f"Query: {query}"}]
    )

    raw = response.content[0].text.strip()
    # strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    data = json.loads(raw)

    subtasks = {}
    for item in data["subtasks"]:
        st = SubTask(
            id=item["id"],
            action_verb=item["action_verb"],
            target_noun=item["target_noun"],
            description=item["description"],
            tool_needs=item.get("tool_needs", [])
        )
        subtasks[st.id] = st

    return subtasks


# ─────────────────────────────────────────────
# Stage 2 — Discover relationships (3 mechanisms)
# ─────────────────────────────────────────────

RELATIONSHIP_SYSTEM = """You are a dependency analysis engine.
Given a list of subtasks, discover ALL relationships between them using three mechanisms:

1. TEMPORAL — ordering language implies one must finish before another starts
   ("search before", "then validate", "once X is done, Y can begin")

2. DATA_FLOW — the output of one subtask is the input of another
   (subtask A produces something that subtask B consumes)

3. SEMANTIC — SRL: A produces the entity/argument that B operates on
   (more subtle than data flow — about argument roles, not just values)

Return ONLY valid JSON — no markdown, no explanation:
{
  "edges": [
    {
      "source": "s1",
      "target": "s2",
      "edge_type": "temporal",
      "label": ""
    },
    {
      "source": "s2",
      "target": "s4",
      "edge_type": "data_flow",
      "label": "extracted label words"
    }
  ]
}

Rules:
- Only add edges where a genuine dependency exists
- A DAG must be acyclic — never create a cycle
- DATA_FLOW edges must name what flows in the label field
- Prefer fewer, high-confidence edges over many speculative ones
"""

def discover_relationships(client: anthropic.Anthropic, subtasks: dict[str, SubTask]) -> list[Edge]:
    """Stage 2: Three-mechanism relationship discovery → edge list."""
    subtask_descriptions = "\n".join(
        f"{st.id}: [{st.action_verb}] {st.target_noun} — {st.description}"
        for st in subtasks.values()
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=RELATIONSHIP_SYSTEM,
        messages=[{"role": "user", "content": subtask_descriptions}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    data = json.loads(raw)

    edges = []
    for item in data["edges"]:
        edges.append(Edge(
            source=item["source"],
            target=item["target"],
            edge_type=EdgeType(item["edge_type"]),
            label=item.get("label", "")
        ))

    return edges


# ─────────────────────────────────────────────
# Stage 3 — Build DAG: compute degrees
# ─────────────────────────────────────────────

def build_dag(query: str, subtasks: dict[str, SubTask], edges: list[Edge]) -> DAG:
    """Stage 3: Assemble DAG and compute in/out degrees from edge list."""
    dag = DAG(subtasks=subtasks, edges=edges, query=query)

    for edge in edges:
        if edge.source in dag.subtasks:
            dag.subtasks[edge.source].out_degree += 1
        if edge.target in dag.subtasks:
            dag.subtasks[edge.target].in_degree += 1

    return dag


# ─────────────────────────────────────────────
# Stage 4 — Score each node: R, T, S, ρ → ASS
# ─────────────────────────────────────────────

SCORE_SYSTEM = """You are a task complexity scoring engine.
Score each subtask on four dimensions (0.0–10.0):

R  — Reasoning depth
     0 = trivial lookup, 5 = multi-step inference, 10 = complex multi-hop reasoning

T  — Tool / resource need
     0 = pure reasoning (no tools), 5 = one external call, 10 = multiple tools required

S  — State / memory need
     0 = stateless, 5 = needs context from one prior step, 10 = must persist data across many steps

rho — Risk / failure probability
     0 = near-certain success, 5 = moderate uncertainty, 10 = high risk of failure/ambiguity

Return ONLY valid JSON — no markdown, no explanation:
{
  "scores": {
    "s1": {"R": 4.0, "T": 9.0, "S": 2.0, "rho": 6.0},
    "s2": {"R": 7.0, "T": 8.0, "S": 5.0, "rho": 7.5}
  }
}
"""

# ASS formula weights (must sum to 1.0)
WEIGHTS = {"R": 0.30, "T": 0.25, "S": 0.25, "rho": 0.20}

def score_nodes(client: anthropic.Anthropic, dag: DAG) -> DAG:
    """Stage 4: LLM scores each node; compute ASS = weighted sum."""

    # Build context: subtask details + structural info from DAG
    context_lines = []
    for st in dag.subtasks.values():
        context_lines.append(
            f"{st.id}: {st.description}\n"
            f"  tools={st.tool_needs}, in_degree={st.in_degree}, out_degree={st.out_degree}"
        )

    response = dag._client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=SCORE_SYSTEM,
        messages=[{"role": "user", "content": "\n".join(context_lines)}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    data = json.loads(raw)

    for sid, scores in data["scores"].items():
        if sid not in dag.subtasks:
            continue
        st = dag.subtasks[sid]
        st.R   = float(scores["R"])
        st.T   = float(scores["T"])
        st.S   = float(scores["S"])
        st.rho = float(scores["rho"])
        st.ASS = (
            WEIGHTS["R"]   * st.R +
            WEIGHTS["T"]   * st.T +
            WEIGHTS["S"]   * st.S +
            WEIGHTS["rho"] * st.rho
        )

    # Aggregate complexity = mean ASS weighted by in_degree+1 (bottleneck nodes matter more)
    total_weight = sum(st.in_degree + 1 for st in dag.subtasks.values())
    dag.aggregate_complexity = sum(
        st.ASS * (st.in_degree + 1) for st in dag.subtasks.values()
    ) / total_weight if total_weight else 0.0

    dag.needs_agent = dag.aggregate_complexity >= 6.0

    return dag


# ─────────────────────────────────────────────
# Render — pretty print results
# ─────────────────────────────────────────────

def render(dag: DAG) -> None:
    EDGE_SYMBOLS = {
        EdgeType.TEMPORAL:  "──►",
        EdgeType.DATA_FLOW: "- ->",
        EdgeType.SEMANTIC:  "═══►",
    }

    print("=" * 64)
    print(f"  QUERY: {dag.query}")
    print("=" * 64)

    print("\n── SUBTASKS ─────────────────────────────────────────────\n")
    for st in dag.subtasks.values():
        print(f"  {st.id}  {st.action_verb} [{st.target_noun}]")
        print(f"      {st.description}")
        print(f"      tools: {st.tool_needs or 'none'}")
        print(f"      in={st.in_degree}, out={st.out_degree}")
        print(f"      R={st.R:.1f}  T={st.T:.1f}  S={st.S:.1f}  ρ={st.rho:.1f}  "
              f"ASS={st.ASS:.2f}")
        print()

    print("── EDGES ────────────────────────────────────────────────\n")
    for e in dag.edges:
        sym = EDGE_SYMBOLS[e.edge_type]
        label = f"  [{e.label}]" if e.label else ""
        print(f"  {e.source} {sym} {e.target}  ({e.edge_type.value}){label}")

    print("\n── AGGREGATE COMPLEXITY ─────────────────────────────────\n")
    bar_len = int(dag.aggregate_complexity / 10 * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    print(f"  [{bar}]  {dag.aggregate_complexity:.2f} / 10.0")
    print(f"  Needs agent:  {'YES ⚡' if dag.needs_agent else 'No'}")
    print("=" * 64)


# ─────────────────────────────────────────────
# Pipeline — wire all stages together
# ─────────────────────────────────────────────

def decompose(query: str) -> DAG:
    """
    Full 4-stage pipeline:
      Stage 1 — Parse query into subtasks (LLM)
      Stage 2 — Discover relationships (LLM, 3 mechanisms)
      Stage 3 — Build DAG (compute degrees)
      Stage 4 — Score each node (LLM), compute ASS
    """
    client = anthropic.Anthropic()

    print(f"\n[Stage 1] Parsing query into subtasks...")
    subtasks = parse_query(client, query)
    print(f"          → {len(subtasks)} subtasks found")

    print("[Stage 2] Discovering relationships...")
    edges = discover_relationships(client, subtasks)
    print(f"          → {len(edges)} edges discovered")

    print("[Stage 3] Building DAG...")
    dag = build_dag(query, subtasks, edges)

    # attach client for Stage 4 (score_nodes needs it)
    dag._client = client

    print("[Stage 4] Scoring nodes...\n")
    dag = score_nodes(client, dag)

    render(dag)
    return dag


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    query = (
        "A paper about AI regulation that was originally submitted to arXiv.org "
        "in June 2022 shows a figure with three axes, where each axis has a label "
        "word at both ends. Which of these words is used to describe a type of "
        "society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"
    )
    decompose(query)
