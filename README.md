# research-work

QCE query complexity + learned router + agent orchestration for GAIA, Hotpot, MuSiQue, and internal QCE splits.

## Quick start

```bash
# Install (editable package + CLI entry points)
uv sync --dev

# Health check (paths + golden router inference)
uv run research-route verify

# Route GAIA eval sample (no agent execution)
uv run research-pipeline --dataset gaia --route-only --limit 5

# Full pipeline with grading (needs OPENAI_API_KEY)
cp .env.example .env   # fill OPENAI_API_KEY
uv run research-pipeline --dataset gaia --grade --limit 3
```

## CLI entry points

| Command | Purpose |
|---------|---------|
| `research-route` | Router train/eval/analyze/benchmark (default: `verify`) |
| `research-pipeline` | Eval orchestrator: features → route → optional agent run |

Legacy equivalents: `uv run python -m routing …`, `uv run python -m orchestrator …`

## Docker

```bash
docker compose build
docker compose run --rm app research-route verify
docker compose run --rm app research-pipeline --dataset gaia --limit 1 --route-only
```

Production overlay (read-only data/models mounts): `deploy/docker-compose.prod.yml`

## Environment variables

See [`.env.example`](.env.example). Key variables:

- `OPENAI_API_KEY` — required for agent runs and QCE decompose
- `RESEARCH_ROUTER_MODEL_PATH` — override primary router `.joblib`
- `RESEARCH_USE_NEW_PATHS=1` — opt-in data layout ([`scripts/setup_data_layout.py`](scripts/setup_data_layout.py))
- `RESEARCH_LOG_JSON=1` — structured JSON logs on stderr
- `RESEARCH_LLM_*` — retry / rate-limit for decompose LLM calls

## Data layout

**Legacy (default):** `datasets/eval_samples/`, `datasets/qce_features/`, `models/router/graph_main/`

**New (opt-in):** `datasets/input/`, `datasets/intermediate/`, `models/router/production/`

Universe B live baselines stay at `results/orchestrator/graph_main_tuned/` in both modes.

Setup symlinks (no data copy):

```bash
uv run python scripts/setup_data_layout.py
export RESEARCH_USE_NEW_PATHS=1
```

## HTTP API (optional)

```bash
uv sync --group api
uv run uvicorn api.main:app --port 8080
# GET /health  POST /route  GET /ready
```

## Development

```bash
uv run pytest tests/ -q
uv run ruff check config routing orchestrator qce api tests
uv run research-route verify
```

Architecture and refactor notes: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md), [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md).

## Model artifacts

Primary router (deploy default): `models/router/graph_main/hgbm_cvec5_emb_soft_kl.joblib`  
Sidecar manifest: same path with `.json` (`version`, `test_regret`, `train_n`).

Promote a new model by updating `RESEARCH_ROUTER_MODEL_PATH` or replacing the production artifact, then run `research-route verify`.
