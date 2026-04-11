import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class StripAnsiFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return ANSI_RE.sub("", super().format(record))


def _default_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"pac1-{stamp}.log"


def configure_logging() -> tuple[logging.Logger, Path]:
    logger = logging.getLogger("pac1")
    if logger.handlers:
        file_handler = next(
            (
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ),
            None,
        )
        existing_path = (
            Path(file_handler.baseFilename) if file_handler is not None else _default_log_path()
        )
        return logger, existing_path

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = Path(os.getenv("PAC1_LOG_FILE") or _default_log_path())
    log_path.parent.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        StripAnsiFormatter("%(asctime)s %(levelname)s %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_path


LOGGER = logging.getLogger("pac1")
