"""Central logging configuration for NeuroFence.

All modules should use `logging.getLogger(__name__)` and rely on this
function having been called once at process start (CLI / API entrypoint).
We never use print() for anything other than final user-facing report output.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging. Idempotent: safe to call multiple times."""
    root = logging.getLogger("neurofence")
    root.setLevel(level.upper())

    if root.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
