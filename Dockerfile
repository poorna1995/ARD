# research_work — agent pipelines, routing, and eval
# Build:  docker build -t research-work .
# Run:    docker compose run --rm app uv run python -m orchestrator.pipeline --help

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH=/app

# System libraries for OCR, PDFs, and occasional native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

WORKDIR /app

# Dependency layer (cache-friendly)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code
COPY . .

# spaCy is used by src/utils/helpers.py (not listed in pyproject yet)
RUN uv pip install "spacy>=3.8" \
    && uv run python -m spacy download en_core_web_sm

ENV PATH="/app/.venv/bin:${PATH}"

# Override with `docker compose run app <cmd>` or `docker run ... <cmd>`
CMD ["uv", "run", "python", "-m", "orchestrator.pipeline", "--help"]
