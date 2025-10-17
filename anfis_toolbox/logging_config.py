"""Logging configuration for ANFIS toolbox."""

import logging
import sys


def setup_logging(level: str = "INFO", format_string: str | None = None) -> None:
    """Setup logging configuration for ANFIS toolbox.

    Parameters:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        format_string (str, optional): Custom format string for log messages.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure the root logger for anfis_toolbox
    logger = logging.getLogger("anfis_toolbox")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def enable_training_logs() -> None:
    """Enable training progress logs with a simple format."""
    setup_logging(level="INFO", format_string="%(message)s")


def disable_training_logs() -> None:
    """Disable training progress logs."""
    logger = logging.getLogger("anfis_toolbox")
    logger.setLevel(logging.WARNING)
