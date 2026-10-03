"""Entry point: ``python -m dashcam_ml <command> [--config config.yaml]``."""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path

from dashcam_ml.cli.commands import COMMAND_HANDLERS, COMMAND_HELP
from dashcam_ml.config.loader import DEFAULT_CONFIG_PATH, load_config
from dashcam_ml.config.schema import DashcamConfig
from dashcam_ml.domain.enums import Command
from dashcam_ml.utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="dashcam_ml", description="AI Dashcam iOS - Python tooling (train, evaluate, Core ML export)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for command in Command:
        subparser: argparse.ArgumentParser = subparsers.add_parser(name=command.value, help=COMMAND_HELP[command])
        subparser.add_argument(
            "--config",
            type=Path,
            default=DEFAULT_CONFIG_PATH,
            help=f"đường dẫn file cấu hình (mặc định: {DEFAULT_CONFIG_PATH})",
        )
    return parser


def _force_utf8_console() -> None:
    """Windows consoles default to cp1252, which cannot print Vietnamese help/log text."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(*, argv: Sequence[str] | None = None) -> int:
    _force_utf8_console()
    arguments: argparse.Namespace = build_parser().parse_args(args=argv)
    config: DashcamConfig = load_config(config_path=arguments.config)
    setup_logging(level=config.project.log_level)
    command: Command = Command(arguments.command)
    COMMAND_HANDLERS[command](config=config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
