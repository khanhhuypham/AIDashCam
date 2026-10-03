"""Shared anti-flicker + cooldown state machine used by FCW and LDW."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

StateT = TypeVar("StateT", bound=Enum)


@dataclass(frozen=True, slots=True)
class DebounceOutput(Generic[StateT]):
    active_state: StateT
    alert_triggered: bool


class AlertDebouncer(Generic[StateT]):
    """Activates a state only after ``min_consecutive_frames`` frames in a row.

    ``ordered_states`` lists alert states from least to most severe (the idle state
    excluded). A raw state counts toward every state at or below its severity, so a
    CRITICAL frame also keeps the CAUTION streak alive. An alert is *triggered*
    (sound + haptic) when a state becomes active and the cooldown has elapsed, or
    when a more severe state becomes active.
    """

    def __init__(
        self,
        *,
        idle_state: StateT,
        ordered_states: Sequence[StateT],
        min_consecutive_frames: int,
        cooldown_s: float,
    ) -> None:
        self._idle_state: StateT = idle_state
        self._ordered_states: tuple[StateT, ...] = tuple(ordered_states)
        self._min_consecutive_frames: int = min_consecutive_frames
        self._cooldown_s: float = cooldown_s
        self._streaks: dict[StateT, int] = {state: 0 for state in self._ordered_states}
        self._last_alert_s: float | None = None
        self._last_alert_rank: int = -1

    def reset(self) -> None:
        self._streaks = {state: 0 for state in self._ordered_states}
        self._last_alert_s = None
        self._last_alert_rank = -1

    def update(self, *, raw_states: Sequence[StateT], timestamp_s: float) -> DebounceOutput[StateT]:
        """``raw_states`` are all states satisfied in this frame (empty = idle)."""
        active_state: StateT = self._idle_state
        active_rank: int = -1
        for rank, state in enumerate(self._ordered_states):
            self._streaks[state] = self._streaks[state] + 1 if state in raw_states else 0
            if self._streaks[state] >= self._min_consecutive_frames:
                active_state = state
                active_rank = rank

        alert_triggered: bool = False
        if active_rank >= 0:
            cooldown_over: bool = (
                self._last_alert_s is None or timestamp_s - self._last_alert_s >= self._cooldown_s
            )
            if cooldown_over or active_rank > self._last_alert_rank:
                alert_triggered = True
                self._last_alert_s = timestamp_s
                self._last_alert_rank = active_rank
        return DebounceOutput(active_state=active_state, alert_triggered=alert_triggered)
