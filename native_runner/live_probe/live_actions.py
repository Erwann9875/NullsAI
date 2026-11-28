from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Mapping, NamedTuple, Sequence

from ..battle_env import _action_touch_delay_bin, _touch_window_elixir_raw
from ..contracts import ActionKind, ActionV1, ObservationV1, TargetKind
from ..training.v4.expert import POLICY_DECISION_TICKS
from .external_source import _observe_with_snapshot_retry as _observe_with_snapshot_retry
from .live_contracts import LiveLadderError
from .live_observation import LiveObservationBuilder
from .live_settings import EXPECTED_PACKAGE, EXPECTED_VERSION, TICK_MS
from .touch_layout import TouchLayout


class AdbTouchController:
    def __init__(
        self,
        adb: Path,
        serial: str,
        layout: TouchLayout,
        *,
        rng: random.Random | None = None,
        placement_input: str = "swipe",
    ) -> None:
        self.command = [str(adb), "-s", serial]
        self.layout = layout
        self.rng = rng or random.SystemRandom()
        if placement_input not in {"drag", "tap", "swipe"}:
            raise ValueError(f"unknown placement input {placement_input!r}")
        self.placement_input = placement_input

    def _run(self, *args: str) -> str:
        result = subprocess.run(self.command + list(args), capture_output=True, text=True, timeout=15)
        output = (result.stdout + result.stderr).strip()
        if result.returncode:
            raise LiveLadderError(output or f"ADB exited {result.returncode}")
        return output

    def verify(self) -> None:
        size = self._run("shell", "wm", "size")
        expected = f"{self.layout.screen_width}x{self.layout.screen_height}"
        if expected not in size:
            raise LiveLadderError(f"Android surface is {size!r}, expected {expected}")
        package = self._run("shell", "dumpsys", "package", EXPECTED_PACKAGE)
        if not re.search(r"(?m)^\s*primaryCpuAbi=arm64-v8a\s*$", package):
            raise LiveLadderError("live Null's Royale requires the ARM64 game package")
        if not re.search(rf"(?m)^\s*versionName={re.escape(EXPECTED_VERSION)}\s*$", package):
            raise LiveLadderError(f"live Null's Royale requires {EXPECTED_PACKAGE} {EXPECTED_VERSION}")

    def _jitter(self, point: tuple[int, int], radius: int) -> tuple[int, int]:
        x = point[0] + self.rng.randint(-radius, radius)
        y = point[1] + self.rng.randint(-radius, radius)
        return (
            min(self.layout.screen_width - 1, max(0, x)),
            min(self.layout.screen_height - 1, max(0, y)),
        )

    def drag(self, start: tuple[int, int], target: tuple[int, int], duration_ms: int) -> None:
        if any(
            not 0 <= x < self.layout.screen_width or not 0 <= y < self.layout.screen_height
            for x, y in (start, target)
        ):
            raise LiveLadderError("refusing an out-of-surface Android drag")
        self._run(
            "shell",
            "input",
            "touchscreen",
            "swipe",
            str(start[0]),
            str(start[1]),
            str(target[0]),
            str(target[1]),
            str(duration_ms),
        )

    def _place_exact(self, hand: tuple[int, int], target: tuple[int, int]) -> None:
        hx, hy = hand
        x, y = target
        if not (0 <= x < self.layout.screen_width and 0 <= y < self.layout.screen_height):
            raise LiveLadderError("placement falls outside the Android surface")
        if self.placement_input == "tap":
            self._run("shell", f"input touchscreen tap {hx} {hy} && sleep 0.05 && input touchscreen tap {x} {y}")
            return

        motion = "input touchscreen motionevent"
        commands = (
            f"{motion} DOWN {hx} {hy}",
            f"{motion} MOVE {(hx + x) // 2} {(hy + y) // 2}",
            f"{motion} MOVE {x} {y}",
            "sleep 0.06",
            f"{motion} UP {x} {y}",
        )
        try:
            self._run("shell", " && ".join(commands))
        except (LiveLadderError, OSError, subprocess.TimeoutExpired):
            try:
                self._run("shell", "input", "touchscreen", "motionevent", "CANCEL", str(x), str(y))
            except (LiveLadderError, OSError, subprocess.TimeoutExpired):
                pass
            raise

    def execute(self, actions: Sequence[ActionV1]) -> None:
        started = time.monotonic()
        for action in actions:
            if action.kind is ActionKind.WAIT:
                continue
            if action.kind is not ActionKind.PLAY_CARD or action.hand_slot is None:
                raise LiveLadderError(f"unsupported live action kind: {action.kind.value}")
            target_time = started + max(0, int(action.execute_offset_ticks)) * TICK_MS / 1000.0
            remaining = target_time - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            hand_center = (round(self.layout.hand_centers_x[action.hand_slot]), round(self.layout.hand_y))
            if self.placement_input != "swipe":
                self._place_exact(hand_center, self.layout.target_pixel(action))
                continue
            start = self._jitter(hand_center, self.layout.hand_jitter_px)
            target = self.layout.target_pixel(action)
            if action.subcell_offset is None:
                target = self._jitter(target, self.layout.release_jitter_px)
            base_duration = self.layout.drag_duration_ms(start, target)
            duration = base_duration + self.rng.randint(
                -self.layout.drag_duration_jitter_ms,
                self.layout.drag_duration_jitter_ms,
            )
            duration = min(self.layout.drag_max_ms, max(self.layout.drag_min_ms, duration))
            self.drag(start, target, duration)


def _placement_for_play(observation: ObservationV1, action: ActionV1) -> Mapping[str, object]:
    own = observation.players[0]
    if (
        action.kind is not ActionKind.PLAY_CARD
        or action.owner != observation.owner
        or action.hand_slot is None
        or action.card_id is None
        or action.target_kind is not TargetKind.GRID
        or action.target_grid is None
    ):
        raise LiveLadderError("touch output received an unsupported decoded action")
    slot = int(action.hand_slot)
    if not 0 <= slot < len(own.hand):
        raise LiveLadderError(f"decoded hand slot {slot} is outside the four-card live hand")
    if int(own.hand[slot]) != int(action.card_id):
        raise LiveLadderError(
            f"decoded card {action.card_id} does not match live hand slot {slot} ({own.hand[slot]})"
        )
    if not observation.action_mask.hand_slots[slot]:
        raise LiveLadderError(f"decoded hand slot {slot} is not legal in the source snapshot")
    placement = observation.action_mask.placement_masks.get(str(slot))
    if not isinstance(placement, Mapping):
        raise LiveLadderError(f"live placement mask for hand slot {slot} is missing")
    return placement


def _target_allowed(rows: object, grid: tuple[int, int]) -> bool:
    x, y = grid
    return (
        isinstance(rows, Sequence)
        and 0 <= y < len(rows)
        and isinstance(rows[y], Sequence)
        and 0 <= x < len(rows[y])
        and bool(rows[y][x])
    )


def validate_touch_actions(observation: ObservationV1, actions: Sequence[ActionV1], *, due_now: bool = False) -> None:
    plays = [action for action in actions if action.kind is ActionKind.PLAY_CARD]
    if len(plays) > 2 or len({play.hand_slot for play in plays}) != len(plays):
        raise LiveLadderError("a decision permits up to two plays from distinct hand slots")
    if any(first.execute_offset_ticks > second.execute_offset_ticks for first, second in zip(plays, plays[1:])):
        raise LiveLadderError("compound plays have reversed execution offsets")
    effective = observation.action_mask.reasons["effective_elixir"]
    window = _touch_window_elixir_raw(observation.tick, round(float(effective) * 10_000))
    spent = 0
    for action in actions:
        if action.kind is ActionKind.WAIT:
            continue
        placement = _placement_for_play(observation, action)
        spent += round(float(placement["effective_cost"]) * 10_000)
        delay_bin = 0 if due_now else _action_touch_delay_bin(action)
        if spent > window[delay_bin] + 0.5:
            raise LiveLadderError("insufficient elixir for the proposed plays at their execution offsets")
        if not _target_allowed(placement.get("row_major"), action.target_grid):
            raise LiveLadderError(f"decoded target {action.target_grid} is illegal for live hand slot {action.hand_slot}")


class ScheduledPlay(NamedTuple):
    action: ActionV1
    due_tick: int
    episode_id: str


class LivePlayDispatcher:
    def __init__(self, builder: LiveObservationBuilder, controller: AdbTouchController) -> None:
        self.builder = builder
        self.controller = controller
        self.follow_up: ScheduledPlay | None = None
        self.submitted = 0

    def _send(self, observation: ObservationV1, cursor: int, action: ActionV1) -> None:
        self.controller.execute((action,))
        self.builder.record_actions((action,), observation.tick, cursor)
        self.submitted += 1

    def submit(self, observation: ObservationV1, cursor: int, actions: Sequence[ActionV1], limit: int | None) -> int:
        if observation.terminal.ended or self.follow_up is not None:
            raise LiveLadderError("cannot start a play on this snapshot")
        validate_touch_actions(observation, actions)
        plays = [action for action in actions if action.kind is ActionKind.PLAY_CARD]
        allowed = plays if limit is None else plays[: max(0, limit - self.submitted)]
        if allowed:
            self._send(observation, cursor, allowed[0])
        if len(allowed) == 2:
            second = allowed[1]
            self.follow_up = ScheduledPlay(second, observation.tick + second.execute_offset_ticks, observation.episode_id)
        return len(allowed)

    def advance(self, observation: ObservationV1, cursor: int) -> str:
        if self.follow_up is None:
            raise LiveLadderError("there is no scheduled follow-up")
        action, due, episode = self.follow_up
        if observation.terminal.ended or observation.episode_id != episode:
            reason = "match ended or changed"
        elif observation.tick > due + POLICY_DECISION_TICKS:
            reason = "decision expired"
        elif observation.tick < due:
            return "scheduled"
        else:
            try:
                validate_touch_actions(observation, (action,), due_now=True)
            except LiveLadderError as error:
                reason = str(error)
            else:
                self._send(observation, cursor, replace(action, execute_offset_ticks=0))
                self.follow_up = None
                return "submitted"
        self.follow_up = None
        return reason
