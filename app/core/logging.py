"""
Structured logging setup for AI Broker.

Usage::

    from app.core.logging import get_logger
    log = get_logger("t212")
    log.info("Fetched positions", extra={"count": 5, "elapsed": 1.23})
"""

from __future__ import annotations

import logging
import sys


_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root ``ai_broker`` logger once.  Safe to call multiple times."""
    global _configured
    if _configured:
        return
    _configured = True

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger("ai_broker")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    # Prevent duplicate output from the default root logger
    root.propagate = False


def get_logger(component: str) -> logging.Logger:
    """Return a child logger under the ``ai_broker`` namespace.

    Example: ``get_logger("t212")`` → ``ai_broker.t212``
    """
    return logging.getLogger(f"ai_broker.{component}")
