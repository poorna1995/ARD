"""Environment / secrets validation (never log secret values)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env when present (local dev / Docker env_file).
load_dotenv()

REQUIRED_FOR_LLM_DECOMPOSE = ("OPENAI_API_KEY",)
REQUIRED_FOR_AGENT_RUNS = ("OPENAI_API_KEY",)  # primary agent backend
OPTIONAL_KEYS = (
    "GROQ_API_KEY",
    "SERPER_API_KEY",
    "SERP_API_KEY",
    "GITHUB_TOKEN",
    "ASSEMBLYAI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "RESEARCH_ROUTER_MODEL_PATH",
    "RESEARCH_OUTPUT_ROOT",
    "RESEARCH_USE_NEW_PATHS",
)


@dataclass(frozen=True)
class SecretCheckResult:
    missing_required: tuple[str, ...]
    present_optional: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_required


def check_secrets(*, require: tuple[str, ...] = ()) -> SecretCheckResult:
    missing = tuple(k for k in require if not os.getenv(k, "").strip())
    present_opt = tuple(k for k in OPTIONAL_KEYS if os.getenv(k, "").strip())
    return SecretCheckResult(missing_required=missing, present_optional=present_opt)


def format_secret_report(result: SecretCheckResult) -> str:
    lines = []
    if result.missing_required:
        lines.append(f"missing required: {', '.join(result.missing_required)}")
    if result.present_optional:
        lines.append(f"optional set: {', '.join(result.present_optional)}")
    return "; ".join(lines) if lines else "secrets OK"
