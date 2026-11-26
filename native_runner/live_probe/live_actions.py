from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Mapping, Sequence

from ..battle_env import _action_touch_delay_bin, _touch_window_elixir_raw
from ..contracts import ActionKind, ActionV1, ObservationV1, TargetKind
from ..training.v4.expert import POLICY_DECISION_TICKS
from .external_memory import ExternalMemoryError
from .external_source import ExternalLiveClient
from .live_contracts import LiveLadderError
from .live_observation import LiveObservationBuilder
from .live_settings import EXPECTED_PACKAGE, EXPECTED_VERSION, TICK_MS


class TouchLayout:
    screen_width: int
    screen_height: int
    arena_left: float
    arena_top: float
    arena_right: float
    arena_bottom: float
    hand_y: float
    hand_centers_x: tuple[float, float, float, float]
    owner_zero_at_screen_bottom: bool = True
    drag_min_ms: int = 260
    drag_max_ms: int = 520
    drag_ms_per_pixel: float = 0.35
    hand_jitter_px: int = 5
    release_jitter_px: int = 4
    drag_duration_jitter_ms: int = 35

    @classmethod
    def load(cls, path: Path) -> "TouchLayout":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise LiveLadderError("touch layout must be a JSON object")
        centers = tuple(float(item) for item in value["hand_centers_x"])
        if len(centers) != 4:
            raise LiveLadderError("touch layout needs four hand centers")
        result = cls(
            screen_width=int(value["screen_width"]),
            screen_height=int(value["screen_height"]),
            arena_left=float(value["arena_left"]),
            arena_top=float(value["arena_top"]),
            arena_right=float(value["arena_right"]),
            arena_bottom=float(value["arena_bottom"]),
            hand_y=float(value["hand_y"]),
            hand_centers_x=centers,
            owner_zero_at_screen_bottom=bool(value.get("owner_zero_at_screen_bottom", True)),
            drag_min_ms=int(value.get("drag_min_ms", 260)),
            drag_max_ms=int(value.get("drag_max_ms", 520)),
            drag_ms_per_pixel=float(value.get("drag_ms_per_pixel", 0.35)),
            hand_jitter_px=int(value.get("hand_jitter_px", 5)),
            release_jitter_px=int(value.get("release_jitter_px", 4)),
            drag_duration_jitter_ms=int(value.get("drag_duration_jitter_ms", 35)),
        )
        if (
            result.screen_width <= 0
            or result.screen_height <= 0
            or not 0 <= result.arena_left < result.arena_right < result.screen_width
            or not 0 <= result.arena_top < result.arena_bottom < result.screen_height
            or not 0 <= result.hand_y < result.screen_height
            or any(not 0 <= center < result.screen_width for center in result.hand_centers_x)
            or result.drag_min_ms <= 0
            or result.drag_max_ms < result.drag_min_ms
            or not math.isfinite(result.drag_ms_per_pixel)
            or result.drag_ms_per_pixel <= 0
            or not 0 <= result.hand_jitter_px <= 12
            or not 0 <= result.release_jitter_px <= 6
            or not 0 <= result.drag_duration_jitter_ms <= 80
        ):
            raise LiveLadderError("touch layout coordinates are outside the Android surface")
        return result

    def target_pixel(self, action: ActionV1) -> tuple[int, int]:
        if action.target_kind is not TargetKind.GRID or action.target_grid is None:
            raise LiveLadderError("live touch output currently supports GRID card actions only")
        x, y = action.target_grid
        offset_x, offset_y = action.subcell_offset or (0.0, 0.0)
        fraction_x = (x + 0.5 + float(offset_x)) / 18.0
        fraction_y = (y + 0.5 + float(offset_y)) / 32.0
        pixel_x = self.arena_left + fraction_x * (self.arena_right - self.arena_left)
        if self.owner_zero_at_screen_bottom:
            pixel_y = self.arena_bottom - fraction_y * (self.arena_bottom - self.arena_top)
        else:
            pixel_y = self.arena_top + fraction_y * (self.arena_bottom - self.arena_top)
        return round(pixel_x), round(pixel_y)

    def drag_duration_ms(self, start: tuple[int, int], target: tuple[int, int]) -> int:
        distance = math.hypot(target[0] - start[0], target[1] - start[1])
        duration = self.drag_min_ms + distance * self.drag_ms_per_pixel
        return round(min(self.drag_max_ms, max(self.drag_min_ms, duration)))

class AdbTouchController:
    def __init__(
        self,
        adb: Path,
        serial: str,
        layout: TouchLayout,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.command = [str(adb), "-s", serial]
        self.layout = layout
        self.rng = rng or random.SystemRandom()

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

def validate_touch_actions(observation: ObservationV1, actions: Sequence[ActionV1], *, due_now: bool = False) -> None:

    own = observation.players[0]
    plays = tuple(action for action in actions if action.kind is ActionKind.PLAY_CARD)
    if len(plays) > 2 or len({a.hand_slot for a in plays}) != len(plays):
        raise LiveLadderError("a decision permits up to two plays from distinct hand slots")
    if any(a.execute_offset_ticks > b.execute_offset_ticks for a, b in zip(plays, plays[1:])):
        raise LiveLadderError("compound plays have reversed execution offsets")
    effective = observation.action_mask.reasons["effective_elixir"]
    window = _touch_window_elixir_raw(observation.tick, round(float(effective) * 10_000))
    spent = 0
    for action in actions:
        if action.kind is ActionKind.WAIT:
            continue
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
        spent += round(float(placement["effective_cost"]) * 10_000)
        delay_bin = 0 if due_now else _action_touch_delay_bin(action)
        if spent > window[delay_bin] + 0.5:
            raise LiveLadderError("insufficient elixir for the proposed plays at their execution offsets")
        rows = placement.get("row_major")
        x, y = action.target_grid
        if (
            not isinstance(rows, Sequence)
            or not 0 <= y < len(rows)
            or not isinstance(rows[y], Sequence)
            or not 0 <= x < len(rows[y])
            or not bool(rows[y][x])
        ):
            raise LiveLadderError(f"decoded target {action.target_grid} is illegal for live hand slot {slot}")

def _observe_with_snapshot_retry(
    client: ExternalLiveClient,
    event_cursor: int,
    *,
    attempts: int = 40,
) -> dict[str, object]:

    if attempts < 1:
        raise ValueError("snapshot attempts must be positive")
    last_error: ExternalMemoryError | None = None
    for _attempt in range(attempts):
        try:
            return client.observe(event_cursor)
        except ExternalMemoryError as error:
            last_error = error
            time.sleep(0.01)
    raise LiveLadderError(
        f"live memory stayed inconsistent across {attempts} consecutive snapshot retries"
    ) from last_error

class LivePlayDispatcher:

    def __init__(self, builder: LiveObservationBuilder, controller: AdbTouchController):
        self.builder = builder
        self.controller = controller
        self.follow_up: tuple[ActionV1, int, str] | None = None
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
        allowed = plays if limit is None else plays[:max(0, limit - self.submitted)]
        if allowed:
            self._send(observation, cursor, allowed[0])
        if len(allowed) == 2:
            second = allowed[1]
            self.follow_up = second, observation.tick + second.execute_offset_ticks, observation.episode_id
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
