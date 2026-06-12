"""Backward-compat shim — full surface of ``config.global_config.runtime`` (LLM retry)."""

import config.global_config.runtime as _impl
import sys

_mod = sys.modules[__name__]
for _name in dir(_impl):
    if not _name.startswith("__"):
        setattr(_mod, _name, getattr(_impl, _name))
