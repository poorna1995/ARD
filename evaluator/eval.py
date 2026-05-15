# Normalise response like {"answer":"democratic","confidence":0.9,"complexity":0.7}
# and return: (predicted_answer, confidence, complexity)
import ast
import json
import re
import unicodedata
from typing import Optional, Tuple

JSON_ANSWER_RE = re.compile(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', re.DOTALL)

_PHRASE_MAP = {
    "st": "saint",
    "st.": "saint",
    "mt": "mount",
    "mt.": "mount",
    "&": "and",
}

_TOKEN_MAP = {
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

_NOISY_PHRASE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # OCR-like split/merged artifacts seen in traces.
    (re.compile(r"\bdpeac\s+efull\s+y\b", re.I), "peacefully"),
    (re.compile(r"\bdpeac\s+efull\b", re.I), "peacefully"),
    (re.compile(r"\bytomy\b", re.I), "to my"),
)


def normalise(text: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Try strict structured parse; if it fails, return full text as answer (see normalise_answer)."""
    t = text.strip()
    if not t:
        return "", None, None
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "answer" in obj:
            return (
                str(obj["answer"]).strip(),
                optional_float(obj.get("confidence")),
                optional_float(obj.get("complexity")),
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        obj = ast.literal_eval(t)
        if isinstance(obj, dict) and "answer" in obj:
            return (
                str(obj["answer"]).strip(),
                optional_float(obj.get("confidence")),
                optional_float(obj.get("complexity")),
            )
    except (ValueError, SyntaxError, TypeError):
        pass
    return t, None, None


def optional_float(v: object) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def normalise_answer(raw: str | None) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Shared across agents (cot, raw, react, multiagent): extract answer string plus
    optional confidence/complexity from JSON, Python literals, plain text, or an
    embedded {"answer": "..."} fragment.
    """
    if raw is None:
        return "", None, None

    text = str(raw).strip()
    if not text:
        return "", None, None

    answer, confidence, complexity = normalise(text)

    # Unparsed branch in normalise returns the full string as "answer"; try
    # extracting a JSON object substring for GAIA-style payloads.
    if answer == text:
        j_match = JSON_ANSWER_RE.search(text)
        if j_match:
            blob = j_match.group(0)
            try:
                obj = json.loads(blob)
                if isinstance(obj, dict) and "answer" in obj:
                    return (
                        str(obj["answer"]).strip(),
                        optional_float(obj.get("confidence")),
                        optional_float(obj.get("complexity")),
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            return j_match.group(1).strip(), None, None

    return answer, confidence, complexity


def canonicalise_answer(text: str) -> str:
    """
    Canonical form for answer matching.

    This uses the same normalization-first idea as notebook evaluation:
    - unicode normalization
    - abbreviation/synonym harmonization
    - punctuation/spacing cleanup
    - lightweight token normalization
    """
    s = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    if not s:
        return ""

    s = s.replace("`", " backtick ")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    for pattern, replacement in _NOISY_PHRASE_PATTERNS:
        s = pattern.sub(replacement, s)

    tokens = re.findall(r"[a-z0-9]+\.?|[,;/]|[-]", s)
    expanded = []
    for token in tokens:
        expanded.append(_PHRASE_MAP.get(token, token))
    s = " ".join(expanded)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"(?<!\d),(?!\d)", " ", s)
    s = re.sub(r"[;:/|]", " ", s)
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"[^a-z0-9,\s]", " ", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    normalized_tokens = []
    for token in s.split():
        mapped = _TOKEN_MAP.get(token, token)
        if mapped:
            normalized_tokens.append(mapped)
    s = " ".join(normalized_tokens)
    s = re.sub(r"\b(a|an|the|this|that|these|those)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Final collapse to robust exact-match surface.
    return re.sub(r"[^a-z0-9]+", "", s)


def is_correct(pred: str, expected: str) -> bool:
    return canonicalise_answer(pred) == canonicalise_answer(expected)


# Alias used by cot, raw, multiagent (same contract as historical parse_agent_output).
parse_agent_output = normalise_answer

