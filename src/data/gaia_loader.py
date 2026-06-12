"""Backward-compat shim → ``input.loaders.gaia_loader``."""

from input.loaders.gaia_loader import GAIALoader, append_attachment_to_query

__all__ = ["GAIALoader", "append_attachment_to_query"]
