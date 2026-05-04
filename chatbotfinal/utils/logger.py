"""
STATOUR Structured Logger
===========================
Single shared logger for all agents and tools.
Outputs to both console (INFO+) and daily log file (DEBUG+).

Usage:
    from utils.logger import get_logger
    logger = get_logger("statour.mymodule")
    logger.info("Something happened")
    logger.error("Failed: %s", error, exc_info=True)
"""

import os
import logging
from datetime import datetime

# Import path — handle both direct run and module import
try:
    from config.settings import LOGS_DIR
except ImportError:
    LOGS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs"
    )
    os.makedirs(LOGS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Logger Registry (prevents duplicate handlers)
# ══════════════════════════════════════════════════════════════════════════════
_loggers_initialized = set()


def get_logger(name: str = "statour") -> logging.Logger:
    """
    Get or create a named logger with console + file output.

    Args:
        name: Logger name (use dotted notation: "statour.agent.normal")

    Returns:
        Configured logging.Logger instance.

    Example:
        logger = get_logger("statour.orchestrator")
        logger.info("Routing to %s", agent_name)
        logger.debug("Full context: %s", context)
        logger.error("LLM call failed: %s", err, exc_info=True)
    """
    logger = logging.getLogger(name)

    # Only configure once per logger name
    if name in _loggers_initialized:
        return logger

    logger.setLevel(logging.DEBUG)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    # ── Console Handler (INFO and above) ──────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ── File Handler (DEBUG and above) ────────────────────────────────────
    try:
        log_filename = f"statour_{datetime.now():%Y%m%d}.log"
        log_filepath = os.path.join(LOGS_DIR, log_filename)

        file_handler = logging.FileHandler(
            log_filepath,
            encoding="utf-8",
            mode="a",  # Append mode
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(funcName)-20s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # If we can't create the log file, continue with console only
        logger.warning("Could not create log file: %s — console logging only", e)

    _loggers_initialized.add(name)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: Module-level logger for quick imports
# ══════════════════════════════════════════════════════════════════════════════
# Usage: from utils.logger import log
#        log.info("Quick message")

log = get_logger("statour")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Logger Test")
    print("=" * 60)
    print()

    test_logger = get_logger("statour.test")

    test_logger.debug("This is a DEBUG message (file only)")
    test_logger.info("This is an INFO message (console + file)")
    test_logger.warning("This is a WARNING message")
    test_logger.error("This is an ERROR message")

    try:
        1 / 0
    except Exception:
        test_logger.error("Exception with traceback:", exc_info=True)

    # Test that duplicate calls don't add extra handlers
    test_logger2 = get_logger("statour.test")
    assert len(test_logger2.handlers) <= 2, "Duplicate handlers detected!"

    print()
    print(f"✅ Log file: {os.path.join(LOGS_DIR, f'statour_{datetime.now():%Y%m%d}.log')}")
    print("✅ All tests passed")