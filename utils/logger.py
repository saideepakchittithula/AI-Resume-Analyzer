# utils/logger.py
"""
logger.py — Centralised logging configuration for AI Resume Analyzer.

Every module calls get_logger(__name__) to get a named logger that
writes to both the console and a rotating log file simultaneously.

Features:
  - Rotating file handler  (5 MB per file, 3 backups)
  - Coloured console output (by log level)
  - Single configuration point — change here, applies everywhere
  - Thread-safe (Python's logging module is thread-safe by default)
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from utils.constants import (
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_FILE,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOGS_DIR,
)

# Ensure the logs directory exists before creating the file handler
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ANSI COLOR CODES FOR CONSOLE OUTPUT
# ============================================================

class _ColorFormatter(logging.Formatter):
    """
    Custom log formatter that adds ANSI color codes to console output
    based on the severity level.  Colors are stripped automatically
    when output is redirected to a file.
    """

    LEVEL_COLORS: dict[int, str] = {
        logging.DEBUG:    "\033[36m",    # Cyan
        logging.INFO:     "\033[32m",    # Green
        logging.WARNING:  "\033[33m",    # Yellow
        logging.ERROR:    "\033[31m",    # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET: str = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Only apply colors when writing to a real terminal
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
            record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


# ============================================================
# LOGGER FACTORY
# ============================================================

# Registry to avoid adding duplicate handlers to the same logger
_configured_loggers: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Retrieve (or create) a named logger with console and file handlers.

    Calling get_logger with the same name multiple times always returns
    the same logger without adding duplicate handlers.

    Args:
        name:  Logger name — use __name__ in every module.
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.

    Example:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing resume...")
        logger.error("Failed to parse PDF: %s", error)
    """
    logger = logging.getLogger(name)

    # Skip reconfiguration if this logger was already set up
    if name in _configured_loggers:
        return logger

    logger.setLevel(level)

    # Prevent log records from bubbling up to the root logger
    # which avoids duplicate output in Streamlit's environment
    logger.propagate = False

    # ----------------------------------------------------------
    # Console handler — coloured, writes to stderr
    # ----------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        _ColorFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )

    # ----------------------------------------------------------
    # Rotating file handler — plain text, no colors
    # ----------------------------------------------------------
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_handler.setFormatter(
            logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as exc:
        # If we can't write the log file, fall back to console-only
        console_handler.setLevel(logging.DEBUG)
        logger.warning("Could not create log file: %s — logging to console only.", exc)

    logger.addHandler(console_handler)
    _configured_loggers.add(name)

    return logger


# ============================================================
# ROOT APPLICATION LOGGER
# ============================================================

# A convenience logger for top-level app messages
app_logger = get_logger("ai_resume_analyzer")


def log_section(title: str) -> None:
    """
    Log a visually distinct section divider for readability in logs.

    Args:
        title: Section title to display between dividers.
    """
    app_logger.info("=" * 60)
    app_logger.info("  %s", title)
    app_logger.info("=" * 60)


def log_success(message: str) -> None:
    """Log a success message with a checkmark prefix."""
    app_logger.info("✅ %s", message)


def log_warning(message: str) -> None:
    """Log a warning message with a warning prefix."""
    app_logger.warning("⚠️  %s", message)


def log_error(message: str) -> None:
    """Log an error message with a cross prefix."""
    app_logger.error("❌ %s", message)
