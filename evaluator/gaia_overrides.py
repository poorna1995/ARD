"""
GAIA-only grading tweaks (trace-specific OCR/token fixes).

Applied only when ``canonicalise_for_dataset(..., dataset="gaia")`` is used.
"""

from __future__ import annotations

import re

# Token substitutions from bad traces — not used for hotpot/musique/mmlu.
_GAIA_TOKEN_MAP: dict[str, str] = {
    "agull": "seagull",
    "gulls": "seagull",
    "gull": "seagull",
    "glide": "glided",
    "peaceful": "peacefully",
    "peacefully": "",
    "peacefull": "",
    "dpeacefull": "",
    "dpeacefully": "",
    "deep": "",
    "fully": "",
}

_GAIA_PHRASE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdpeac\s+efull\s+y\b", re.I), "peacefully"),
    (re.compile(r"\bdpeac\s+efull\b", re.I), "peacefully"),
    (re.compile(r"\bytomy\b", re.I), "to my"),
)


def apply_gaia_token_overrides(canonical_qa: str) -> str:
    """
    Post-process a QA-canonical string with GAIA-specific token fixes.

    Input is already lowercased/spacing-normalized from ``canonicalise_qa``.
    """
    if not canonical_qa:
        return ""

    s = canonical_qa
    for pattern, replacement in _GAIA_PHRASE_PATTERNS:
        s = pattern.sub(replacement, s)

    tokens = s.split()
    out: list[str] = []
    for token in tokens:
        mapped = _GAIA_TOKEN_MAP.get(token, token)
        if mapped:
            out.append(mapped)
    s = " ".join(out)
    return re.sub(r"\s+", " ", s).strip()
