from __future__ import annotations

from dataclasses import dataclass
import re
import struct

from .game_profile import (
    AUTHORIZATION_END_EPOCH_EXCLUSIVE as AUTHORIZATION_END_UTC_EXCLUSIVE,
    EXPECTED_LIBG_BUILD_ID,
    EXPECTED_LIBG_SHA256,
    EXPECTED_PACKAGE as PACKAGE,
    HOG_26_DECK,
    HOG_26_SET,
)

TOKEN = "nullsroyale-owner-ladder-2026-11-23_2026-12-01"

MAX_TRANSFER = 4 * 1024 * 1024
SCAN_OVERLAP = 0x100
MAX_WORLD_CANDIDATES = 8_192
MAX_OBJECTS = 256
MAX_COMPONENTS = 64
MAX_EFFECTS = 64
MAX_SNAPSHOT_ATTEMPTS = 3
MAX_SNAPSHOT_TICK_SPAN = 4
WORLD_LOCAL_IDENTITY = 0x30
WORLD_OPPONENT_IDENTITY = 0x38
WORLD_LOCAL_DECK = 0x88
WORLD_OPPONENT_DECK = 0x90
WORLD_LOCAL_ROOT = 0xE0
WORLD_OPPONENT_ROOT = 0xE8
WORLD_RESULT = 0x1B8
WORLD_FINALIZED = 0x1E0
MANAGER_TICK = 0x60
MANAGER_WORLD = 0xA8
ROOT_OBJECT_MANAGER = 0x10
ROOT_HAND_VECTOR = 0x210
ROOT_CYCLE_VECTOR = 0x220
ROOT_ELIXIR = 0x2F8
DECK_SLOT_VECTOR = 0x20
ARENA_NATIVE_HEIGHT = 32_000

MAP_PATTERN = re.compile(
    r"^(?P<start>[0-9a-f]+)-(?P<end>[0-9a-f]+)\s+"
    r"(?P<permissions>[r-][w-][x-][ps])\s+(?P<offset>[0-9a-f]+)\s+"
    r"\S+\s+\d+\s*(?P<name>.*)$"
)


class ExternalMemoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessMapping:
    start: int
    end: int
    permissions: str
    file_offset: int
    name: str

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def readable(self) -> bool:
        return self.permissions.startswith("r")

    @property
    def writable(self) -> bool:
        return len(self.permissions) > 1 and self.permissions[1] == "w"

    def contains(self, address: int, size: int = 1) -> bool:
        return size > 0 and self.start <= address and address + size <= self.end


@dataclass(frozen=True, slots=True)
class NativeVector:
    data: int
    capacity: int
    count: int


@dataclass(frozen=True, slots=True)
class BattleLayout:
    pid: int
    libg_base: int
    manager: int
    world: int
    local_identity: int
    local_deck: int
    local_root: int
    opponent_deck: int
    opponent_root: int
    object_manager: int
    account_id: int
    deck: tuple[int, ...]
    native_local_owner: int


def parse_maps(payload: bytes | str) -> tuple[ProcessMapping, ...]:
    text = payload.decode("utf-8", errors="strict") if isinstance(payload, bytes) else payload
    result: list[ProcessMapping] = []
    for line in text.splitlines():
        match = MAP_PATTERN.fullmatch(line.strip())
        if match is None:
            continue
        result.append(
            ProcessMapping(
                start=int(match.group("start"), 16),
                end=int(match.group("end"), 16),
                permissions=match.group("permissions"),
                file_offset=int(match.group("offset"), 16),
                name=match.group("name").strip(),
            )
        )
    if not result:
        raise ExternalMemoryError("reader returned no parseable process mappings")
    return tuple(result)


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _vector(data: bytes, offset: int, maximum: int) -> NativeVector | None:
    pointer = _u64(data, offset)
    capacity = _i32(data, offset + 8)
    count = _i32(data, offset + 12)
    if count < 0 or capacity < count or capacity > maximum or (count > 0 and pointer == 0):
        return None
    return NativeVector(pointer, capacity, count)
