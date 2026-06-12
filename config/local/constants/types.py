"""Shared typing aliases for configuration."""

from __future__ import annotations

from typing import Literal

EvidenceMode = Literal["provided_context", "open_web", "internal"]

__all__ = ["EvidenceMode"]
