from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .live_contracts import LiveLadderError

if TYPE_CHECKING:
    from ..contracts import ActionV1


_OPTIONAL_FIELDS = {
    "owner_zero_at_screen_bottom": bool,
    "drag_min_ms": int,
    "drag_max_ms": int,
    "drag_ms_per_pixel": float,
    "hand_jitter_px": int,
    "release_jitter_px": int,
    "drag_duration_jitter_ms": int,
}


@dataclass(frozen=True, slots=True)
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
        return cls(
            screen_width=int(value["screen_width"]),
            screen_height=int(value["screen_height"]),
            arena_left=float(value["arena_left"]),
            arena_top=float(value["arena_top"]),
            arena_right=float(value["arena_right"]),
            arena_bottom=float(value["arena_bottom"]),
            hand_y=float(value["hand_y"]),
            hand_centers_x=centers,
            **{key: convert(value[key]) for key, convert in _OPTIONAL_FIELDS.items() if key in value},
        )

    def __post_init__(self) -> None:
        valid_surface = (
            0 <= self.arena_left < self.arena_right < self.screen_width
            and 0 <= self.arena_top < self.arena_bottom < self.screen_height
            and 0 <= self.hand_y < self.screen_height
            and len(self.hand_centers_x) == 4
            and all(0 <= center < self.screen_width for center in self.hand_centers_x)
        )
        valid_drag = (
            0 < self.drag_min_ms <= self.drag_max_ms
            and math.isfinite(self.drag_ms_per_pixel)
            and self.drag_ms_per_pixel > 0
            and 0 <= self.hand_jitter_px <= 12
            and 0 <= self.release_jitter_px <= 6
            and 0 <= self.drag_duration_jitter_ms <= 80
        )
        if not valid_surface or not valid_drag:
            raise LiveLadderError("touch layout coordinates are outside the Android surface")

    def target_pixel(self, action: ActionV1) -> tuple[int, int]:
        from ..contracts import TargetKind

        if action.target_kind is not TargetKind.GRID or action.target_grid is None:
            raise LiveLadderError("live touch output currently supports GRID card actions only")
        return self.grid_pixel(action.target_grid, action.subcell_offset or (0.0, 0.0))

    def grid_pixel(self, grid: tuple[int, int], offset: tuple[float, float] = (0.0, 0.0)) -> tuple[int, int]:
        x, y = grid
        offset_x, offset_y = offset
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
