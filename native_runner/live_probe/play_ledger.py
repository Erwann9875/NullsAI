from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts import ActionV1


@dataclass(frozen=True, slots=True)
class PendingTouch:
    action: ActionV1
    cost: float
    requested_tick: int
    expected_tick: int
    source_cursor: int
    slot_changed: bool = False
    event_seen: bool = False

    def observe(self, hand: Sequence[int], events: Sequence[Mapping[str, object]]) -> PendingTouch:
        action = self.action
        assert action.hand_slot is not None
        rotated = self.slot_changed or hand[action.hand_slot] != action.card_id
        seen = self.event_seen or any(
            event["owner"] == 0
            and event["cardId"] == action.card_id
            and int(event["sequence"]) > self.source_cursor
            and int(event["tick"]) >= self.requested_tick
            for event in events
        )
        return replace(self, slot_changed=rotated, event_seen=seen)

    def outcome(self, tick: int, ended: bool, timeout_ticks: int) -> str | None:
        if self.slot_changed and self.event_seen:
            return "confirmed"
        if ended:
            return "match_ended"
        if tick >= self.expected_tick + timeout_ticks:
            return "timed_out"
        return None


class PlayLedger:
    def __init__(self, execution_delay_ticks: int, receipt_timeout_ticks: int) -> None:
        self.execution_delay_ticks = execution_delay_ticks
        self.receipt_timeout_ticks = receipt_timeout_ticks
        self.pending: list[PendingTouch] = []
        self.receipts: list[tuple[PendingTouch, str]] = []

    def settle(self, tick: int, ended: bool, hand: Sequence[int], events: Sequence[Mapping[str, object]]) -> None:
        remaining = []
        self.receipts.clear()
        for submitted in self.pending:
            observed = submitted.observe(hand, events)
            outcome = observed.outcome(tick, ended, self.receipt_timeout_ticks)
            if outcome is None:
                remaining.append(observed)
            else:
                self.receipts.append((submitted, outcome))
        self.pending[:] = remaining

    def record(self, action: ActionV1, cost: float, requested_tick: int, source_cursor: int) -> None:
        offset = max(1, int(action.execute_offset_ticks or 1))
        self.pending.append(
            PendingTouch(
                action=action,
                cost=cost,
                requested_tick=requested_tick,
                expected_tick=requested_tick + offset + self.execution_delay_ticks,
                source_cursor=source_cursor,
            )
        )
