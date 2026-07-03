"""
VisionTrack Logger Module

Centralised logging configuration with support for:
- Timestamped log files written to logs/
- Colour-coded console output (DEBUG/INFO/WARNING/ERROR)
- Third-party library verbosity control (ultralytics, cv2, torch, etc.)
- A named 'visiontrack' application logger used by all pipeline modules

Usage:
    from loggers import get_logger

    log = get_logger(__name__)
    log.info("Pipeline started")
    log.warning("Low confidence detections on frame 42")
    log.error("Failed to open video file")
"""

import logging
import os
from datetime import datetime
from pathlib import Path

__all__ = [
    "get_logger",
    "configure_logging",
    "LOG_PATH",
]

# ==============================================================================
# ANSI colour codes for console output
# ==============================================================================

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREY = "\033[90m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BRED = "\033[1;91m"  # bold red — ERROR / CRITICAL

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: _GREY,
    logging.INFO: _GREEN,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _BRED,
}


class _ColorFormatter(logging.Formatter):
    """Logging formatter that injects ANSI colour codes around the level name."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, _RESET)
        record.levelname = f"{color}{_BOLD}{record.levelname:<8}{_RESET}"
        return super().format(record)


# ==============================================================================
# Log directory and file setup
# ==============================================================================

try:
    _repo_root = Path(__file__).resolve().parents[1]
except Exception:
    _repo_root = Path(os.getcwd()).resolve()

_LOG_DIR = _repo_root / "logs"

try:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    LOG_PATH: str = str(_LOG_DIR / _LOG_FILE)
except Exception:
    # Never crash the application because logging can't write to disk.
    LOG_PATH = ""

# ==============================================================================
# Root logging configuration
# ==============================================================================

_LOG_LEVEL_NAME: str = (os.getenv("VISIONTRACK_LOG_LEVEL") or "INFO").upper()
_LOG_LEVEL: int = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)

# Plain format for file handler (no ANSI codes)
_FILE_FORMAT = "[%(asctime)s] %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
# Slightly shorter format for console (colour formatter wraps levelname)
_CONSOLE_FORMAT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Apply root config once at module import
_handlers: list[logging.Handler] = []

# -- File handler --
if LOG_PATH:
    try:
        _fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        _fh.setLevel(_LOG_LEVEL)
        _fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        _handlers.append(_fh)
    except Exception:
        pass  # Silently degrade to console-only if file can't be opened

# -- Console handler (colour) --
_ch = logging.StreamHandler()
_ch.setLevel(_LOG_LEVEL)
_ch.setFormatter(_ColorFormatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
_handlers.append(_ch)

logging.basicConfig(level=_LOG_LEVEL, handlers=_handlers)


# ==============================================================================
# Third-party library verbosity control
# ==============================================================================


def configure_logging() -> None:
    """Suppress noisy third-party loggers and apply VisionTrack log level.

    Called automatically on module import. Can be called again after the
    application config is loaded to honour the ``logging.level`` YAML setting.
    """
    # --- Ultralytics / YOLO ---
    # Ultralytics prints its own banner and progress; we want to control this.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # --- PyTorch ---
    logging.getLogger("torch").setLevel(logging.WARNING)

    # --- OpenCV (if it ever logs) ---
    logging.getLogger("cv2").setLevel(logging.WARNING)

    # --- HTTP clients (used by Ultralytics for weight downloads) ---
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # --- Async runtime ---
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # --- PIL / Pillow (used by Ultralytics) ---
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # --- Matplotlib (used for optional plots) ---
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Run on first import
configure_logging()


# ==============================================================================
# Application logger factory
# ==============================================================================


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a named logger under the 'visiontrack' namespace.

    All pipeline modules should obtain their logger via this function so that:
    - Log output is consistent across the application.
    - The log level can be changed centrally via the VISIONTRACK_LOG_LEVEL
      environment variable or by calling configure_logging() again.

    Args:
        name: Typically ``__name__`` of the calling module.
              The prefix 'visiontrack.' is prepended automatically if the
              name doesn't already start with it.
        level: Override log level for this specific logger.

    Returns:
        A configured ``logging.Logger`` instance.

    Example:
        log = get_logger(__name__)
        log.info("Detector initialised with model yolov8n")
    """
    # Normalise name to 'visiontrack.<module>' namespace
    if not name.startswith("visiontrack"):
        # Strip leading 'src.' prefix that appears when modules inside src/
        # call get_logger(__name__) → "src.detector" → "visiontrack.detector"
        short = name.removeprefix("src.").removeprefix("loggers.").removeprefix("exceptions.")
        qualified = f"visiontrack.{short}"
    else:
        qualified = name

    log = logging.getLogger(qualified)
    log.setLevel(level if level is not None else _LOG_LEVEL)
    log.propagate = True
    return log
