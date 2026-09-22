"""Application logger."""

import logging
import sys
import warnings
from pathlib import Path

from loguru import logger
from transformers.utils import logging as transformers_logging

# -------------------------------------------------------------------
# Configure third-party libraries
# -------------------------------------------------------------------

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

transformers_logging.set_verbosity_error()

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Configure Loguru
# -------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)

logger.add(
    LOG_DIR / "application.log",
    level="INFO",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
)

logger.add(
    LOG_DIR / "errors.log",
    level="ERROR",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)

__all__ = ["logger"]