"""Simple logging helpers used by finetune.py."""

from __future__ import annotations

import logging
import os
from datetime import datetime


def log_creater(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("rlhf_asr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class StreamToLogger:
    """File-like object that redirects writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass
