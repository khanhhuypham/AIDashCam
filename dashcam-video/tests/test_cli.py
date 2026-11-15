from __future__ import annotations

import argparse

from dashcam_video.__main__ import build_parser
from dashcam_video.cli.commands import COMMANDS
from dashcam_video.domain.enums import Command


def test_every_command_has_a_handler_and_help() -> None:
    assert set(COMMANDS) == set(Command)
    assert all(spec.help.strip() for spec in COMMANDS.values())


def test_parser_accepts_every_command() -> None:
    parser: argparse.ArgumentParser = build_parser()
    for command in Command:
        assert parser.parse_args(args=[command.value]).command == command.value
