"""One handler per :class:`Command`. Heavy libraries are imported inside handlers so
``--help`` starts instantly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dashcam_video.config.schema import RootConfig
from dashcam_video.domain.enums import Command
from dashcam_video.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)


class CommandHandler(Protocol):
    def __call__(self, *, config: RootConfig) -> None: ...


def run(*, config: RootConfig) -> None:
    from dashcam_video.pipeline.video_runner import run_video

    run_video(config=config)


def export_app_config(*, config: RootConfig) -> None:
    from dashcam_video.export.app_config import export_app_configs

    written: list[Path] = export_app_configs(config=config)
    LOGGER.info("copy each DashcamConfig.json into its app project (Xcode / Android Studio): %d files", len(written))


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """What one CLI command runs and the text ``--help`` shows for it."""

    handler: CommandHandler
    help: str


COMMANDS: dict[Command, CommandSpec] = {
    Command.RUN: CommandSpec(handler=run, help="chạy detect + tracking + FCW (+ LDW) trên video"),
    Command.EXPORT_APP_CONFIG: CommandSpec(
        handler=export_app_config, help="xuất ngưỡng cảnh báo ra DashcamConfig.json cho app iOS và app Android"
    ),
}
