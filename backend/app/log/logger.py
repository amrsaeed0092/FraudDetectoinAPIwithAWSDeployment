"""Structured application logging."""

import logging

from pythonjsonlogger.json import JsonFormatter


def configure_logging(log_level: str) -> None:
    """Configure one JSON console handler without duplicating handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root_logger.addHandler(handler)
