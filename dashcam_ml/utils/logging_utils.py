from __future__ import annotations

import logging

from dashcam_ml.domain.enums import LogLevel

LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(*, level: LogLevel) -> None:
    logging.basicConfig(level=level.value, format=LOG_FORMAT, datefmt="%H:%M:%S", force=True)


def get_logger(*, name: str) -> logging.Logger:
    return logging.getLogger(name=name)
