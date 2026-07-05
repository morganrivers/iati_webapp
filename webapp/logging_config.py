"""Single source of truth for webapp logging configuration."""
import logging
import os

_CONFIGURED = False


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging once. Idempotent; safe to call repeatedly.

    Level resolves from the argument, then the LOG_LEVEL env var, then INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True
