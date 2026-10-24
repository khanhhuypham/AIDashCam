"""One handler per :class:`Command`. Heavy libraries are imported inside handlers so
``--help`` starts instantly."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dashcam_video.config.schema import DashcamConfig
from dashcam_video.domain.enums import Command
from dashcam_video.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)


class CommandHandler(Protocol):
    def __call__(self, *, config: DashcamConfig) -> None: ...


def run(*, config: DashcamConfig) -> None:
    from dashcam_video.pipeline.video_runner import run_video
    run_video(config=config)


def export_ios_config(*, config: DashcamConfig) -> None:
    from dashcam_video.export.ios_config import export_ios_config as write_ios_config

    output_path: Path = write_ios_config(config=config)
    LOGGER.info("copy %s into the Xcode project", output_path.name)


COMMAND_HANDLERS: dict[Command, CommandHandler] = {
    Command.RUN: run,
    Command.EXPORT_IOS_CONFIG: export_ios_config,
}

COMMAND_HELP: dict[Command, str] = {
    Command.RUN: "chạy detect + tracking + FCW (+ LDW) trên video",
    Command.EXPORT_IOS_CONFIG: "xuất ngưỡng cảnh báo ra DashcamConfig.json cho app Swift",
}
