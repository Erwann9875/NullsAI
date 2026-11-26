from __future__ import annotations

from dataclasses import replace
import struct
import time
from typing import Iterable, Sequence

import numpy as np

from .memory_types import (
    DECK_SLOT_VECTOR,
    HOG_26_SET,
    MANAGER_TICK,
    MANAGER_WORLD,
    MAX_TRANSFER,
    MAX_WORLD_CANDIDATES,
    ROOT_CYCLE_VECTOR,
    ROOT_ELIXIR,
    ROOT_HAND_VECTOR,
    ROOT_OBJECT_MANAGER,
    SCAN_OVERLAP,
    WORLD_FINALIZED,
    WORLD_LOCAL_DECK,
    WORLD_LOCAL_IDENTITY,
    WORLD_LOCAL_ROOT,
    WORLD_OPPONENT_DECK,
    WORLD_OPPONENT_IDENTITY,
    WORLD_OPPONENT_ROOT,
    WORLD_RESULT,
    BattleLayout,
    ExternalMemoryError,
    ProcessMapping,
    _i32,
    _u64,
    _vector,
)


class BattleLayoutDiscoveryMixin:
    @staticmethod
    def _account_id(identity: bytes) -> int:
        high, low = struct.unpack_from("<II", identity)
        return (high << 32) | low

    def _libg_base(self) -> int:
        candidates = {
            mapping.start
            for mapping in self.maps
            if mapping.name.endswith("/lib/arm64/libg.so") and mapping.readable and mapping.file_offset == 0
        }
        if len(candidates) != 1:
            raise ExternalMemoryError(f"expected one ARM64 libg base, found {sorted(candidates)}")
        return candidates.pop()

    def _mapped(self, address: int, size: int = 1, *, writable: bool = False) -> bool:
        return any(
            mapping.readable and (not writable or mapping.writable) and mapping.contains(address, size)
            for mapping in self.maps
        )

    def _heap_maps(self) -> tuple[ProcessMapping, ...]:
        result = tuple(
            mapping
            for mapping in self.maps
            if mapping.readable
            and mapping.writable
            and mapping.name in {"[anon:scudo:primary]", "[anon:scudo:secondary]"}
        )
        if not result:
            raise ExternalMemoryError("MuMu exposes no readable native scudo heap")
        return result

    def _chunks(self, mappings: Iterable[ProcessMapping]) -> Iterable[tuple[int, bytes]]:
        step = MAX_TRANSFER - SCAN_OVERLAP
        for mapping in mappings:
            cursor = mapping.start
            while cursor < mapping.end:
                size = min(MAX_TRANSFER, mapping.end - cursor)
                try:
                    payload = self.memory.read(cursor, size)
                except ExternalMemoryError as error:
                    if str(error) != "read range is not contained in a readable mapping":
                        raise


                    break
                yield cursor, payload
                if cursor + size >= mapping.end:
                    break
                cursor += step

    def _world_candidates(self) -> tuple[int, ...]:
        heap = self._heap_maps()
        heap_starts = np.asarray([mapping.start for mapping in heap], dtype=np.uint64)
        heap_ends = np.asarray([mapping.end for mapping in heap], dtype=np.uint64)
        found: set[int] = set()
        primary = tuple(mapping for mapping in heap if mapping.name.endswith(":primary]"))
        phases = (primary, tuple(mapping for mapping in heap if mapping not in primary))
        for phase in phases:
            for address, payload in self._chunks(phase):
                payload = payload[: len(payload) - len(payload) % 8]
                words = np.frombuffer(payload, dtype="<u8")
                count = len(words) - 61
                if count <= 0:
                    continue
                intervals = np.searchsorted(heap_starts, words, side="right") - 1
                safe_intervals = np.maximum(intervals, 0)
                pointers = (intervals >= 0) & (words < heap_ends[safe_intervals])
                mask = (
                    pointers[6 : 6 + count]
                    & pointers[7 : 7 + count]
                    & pointers[17 : 17 + count]
                    & pointers[18 : 18 + count]
                    & pointers[28 : 28 + count]
                    & pointers[29 : 29 + count]
                )
                mask &= words[6 : 6 + count] != words[7 : 7 + count]
                mask &= words[17 : 17 + count] != words[18 : 18 + count]
                mask &= words[28 : 28 + count] != words[29 : 29 + count]
                world_state = words[3 : 3 + count] & np.uint64(0xFFFF_FFFF)
                result_raw = words[55 : 55 + count] & np.uint64(0xFFFF_FFFF)
                finalized = words[60 : 60 + count] & np.uint64(0xFF)
                mask &= world_state <= 16
                mask &= np.isin(result_raw, np.asarray((0, 1, 0xFFFF_FFFF), dtype=np.uint64))
                mask &= finalized <= 1
                aligned = ((address + np.arange(count, dtype=np.uint64) * 8) & 0xF) == 0
                for index in np.flatnonzero(mask & aligned):
                    found.add(address + int(index) * 8)
                    if len(found) > MAX_WORLD_CANDIDATES:
                        raise ExternalMemoryError("native world scan exceeded its candidate bound")
        return tuple(sorted(found))

    def _manager_candidates(self) -> tuple[tuple[int, int], ...]:

        return self.memory.scan_game_states()

    def _select_manager(self, candidates: Sequence[tuple[int, int]]) -> tuple[int, int]:
        if not candidates:
            raise ExternalMemoryError("no active GameState manager matches the current-build invariants")

        def read_candidates() -> tuple[bytes, ...]:
            output: list[bytes] = []
            for start in range(0, len(candidates), 512):
                output.extend(
                    self.memory.readv(
                        tuple((manager, 0x228) for manager, _ in candidates[start : start + 512])
                    )
                )
            return tuple(output)

        before = read_candidates()
        time.sleep(0.25)
        after = read_candidates()
        advancing: list[tuple[int, int]] = []
        stable: list[tuple[int, int]] = []
        for (manager, world), first, second in zip(candidates, before, after, strict=True):
            first_tick = _i32(first, MANAGER_TICK)
            second_tick = _i32(second, MANAGER_TICK)
            if (
                _u64(first, MANAGER_WORLD) != world
                or _u64(second, MANAGER_WORLD) != world
                or not 0 <= first_tick <= second_tick <= first_tick + 50
            ):
                continue
            (advancing if second_tick > first_tick else stable).append((manager, world))
        if len(advancing) == 1:
            return advancing[0]
        if len(advancing) > 1:
            plausible: list[tuple[int, int]] = []
            readable = [item for item in advancing if self._mapped(item[1], 0x200)]
            worlds = self.memory.readv(tuple((world, 0x200) for _, world in readable))
            for (manager, world), data in zip(readable, worlds, strict=True):
                if (
                    0 <= _i32(data, 0x18) <= 16
                    and _i32(data, WORLD_RESULT) in {-1, 0, 1}
                    and int(data[WORLD_FINALIZED]) in {0, 1}
                ):
                    plausible.append((manager, world))
            if len(plausible) == 1:
                return plausible[0]
        raise ExternalMemoryError(
            f"expected one active GameState manager, found advancing={len(advancing)} stable={len(stable)}"
        )

    def discover_manager_world(self) -> tuple[int, int]:
        return self._select_manager(self._manager_candidates())

    def _deck_ids(self, deck: int) -> tuple[int, ...] | None:
        if not self._mapped(deck, 0x30):
            return None
        header = self.memory.read(deck, 0x30)
        slots = _vector(header, DECK_SLOT_VECTOR, 32)
        if slots is None or slots.count != 8 or not self._mapped(slots.data, 64):
            return None
        slot_pointers = struct.unpack("<8Q", self.memory.read(slots.data, 64))
        if any(pointer == 0 or not self._mapped(pointer, 0x18) for pointer in slot_pointers):
            return None
        slot_data = self.memory.readv(tuple((pointer, 0x18) for pointer in slot_pointers))
        card_pointers = tuple(_u64(data, 0x10) for data in slot_data)
        if any(pointer == 0 or not self._mapped(pointer, 0x44) for pointer in card_pointers):
            return None
        card_data = self.memory.readv(tuple((pointer, 0x44) for pointer in card_pointers))
        ids = tuple(_i32(data, 0x40) for data in card_data)
        return ids if all(card > 0 for card in ids) else None

    def _validate_world(self, world: int) -> BattleLayout | None:
        if not self._mapped(world, 0x200):
            return None
        data = self.memory.read(world, 0x200)
        identities = (_u64(data, WORLD_LOCAL_IDENTITY), _u64(data, WORLD_OPPONENT_IDENTITY))
        if any(not self._mapped(identity, 8) for identity in identities):
            return None
        identity_headers = self.memory.readv(tuple((identity, 8) for identity in identities))
        matches = [
            owner
            for owner, identity in enumerate(identity_headers)
            if self._account_id(identity) == self.local_account_id
        ]
        if len(matches) != 1:
            return None
        native_local_owner = matches[0]
        native_opponent_owner = 1 - native_local_owner
        decks = (_u64(data, WORLD_LOCAL_DECK), _u64(data, WORLD_OPPONENT_DECK))
        roots = (_u64(data, WORLD_LOCAL_ROOT), _u64(data, WORLD_OPPONENT_ROOT))
        local_identity = identities[native_local_owner]
        local_deck = decks[native_local_owner]
        opponent_deck = decks[native_opponent_owner]
        local_root = roots[native_local_owner]
        opponent_root = roots[native_opponent_owner]
        ranges = ((local_deck, 0x30), (opponent_deck, 0x30), (local_root, 0x310))
        if any(not self._mapped(address, size) for address, size in ranges):
            return None
        local_deck_data, opponent_deck_data, root = self.memory.readv(ranges)
        own_slots = _vector(local_deck_data, DECK_SLOT_VECTOR, 32)
        enemy_slots = _vector(opponent_deck_data, DECK_SLOT_VECTOR, 32)
        hand = _vector(root, ROOT_HAND_VECTOR, 16)
        cycle = _vector(root, ROOT_CYCLE_VECTOR, 32)
        elixir = _i32(root, ROOT_ELIXIR)
        if (
            own_slots is None
            or own_slots.count != 8
            or enemy_slots is None
            or enemy_slots.count != 8
            or hand is None
            or hand.count > 4
            or cycle is None
            or cycle.count > 8
            or not 0 <= elixir <= 100_000
        ):
            return None
        deck = self._deck_ids(local_deck)
        if deck is None or frozenset(deck) != HOG_26_SET:
            return None
        object_manager = _u64(root, ROOT_OBJECT_MANAGER)
        if not self._mapped(object_manager, 0x18):
            return None
        return BattleLayout(
            pid=self.pid,
            libg_base=self.libg_base,
            manager=0,
            world=world,
            local_identity=local_identity,
            local_deck=local_deck,
            local_root=local_root,
            opponent_deck=opponent_deck,
            opponent_root=opponent_root,
            object_manager=object_manager,
            account_id=self.local_account_id,
            deck=deck,
            native_local_owner=native_local_owner,
        )

    def diagnose_assumed_world(self, world: int) -> dict[str, object]:

        layout = self._validate_world(world)
        if layout is None:
            return {"world": hex(world), "valid": False}
        data = self.memory.read(world, 0x200)
        opponent_header = self.memory.read(layout.opponent_deck, 0x30)
        opponent_slots = _vector(opponent_header, DECK_SLOT_VECTOR, 32)
        root = self.memory.read(layout.local_root, 0x310)
        hand = _vector(root, ROOT_HAND_VECTOR, 16)
        cycle = _vector(root, ROOT_CYCLE_VECTOR, 32)
        report: dict[str, object] = {
            "world": hex(world),
            "valid": True,
            "nativeLocalOwner": layout.native_local_owner,
            "worldState": _i32(data, 0x18),
            "worldResultRaw": _i32(data, WORLD_RESULT),
            "finalized": int(data[WORLD_FINALIZED]),
            "localDeckMatchesHog26": frozenset(layout.deck) == HOG_26_SET,
            "localDeckIds": layout.deck,
            "opponentDeckHeader": None
            if opponent_slots is None
            else {
                "data": hex(opponent_slots.data),
                "capacity": opponent_slots.capacity,
                "count": opponent_slots.count,
            },
            "localRoot": {
                "objectManager": hex(_u64(root, ROOT_OBJECT_MANAGER)),
                "elixirRaw": _i32(root, ROOT_ELIXIR),
                "hand": None
                if hand is None
                else {"data": hex(hand.data), "capacity": hand.capacity, "count": hand.count},
                "cycle": None
                if cycle is None
                else {"data": hex(cycle.data), "capacity": cycle.capacity, "count": cycle.count},
            },
        }
        return report

    def _validate_worlds(self, candidates: Sequence[int]) -> tuple[BattleLayout, ...]:

        valid: list[BattleLayout] = []
        for world in candidates:
            layout = self._validate_world(world)
            if layout is not None:
                valid.append(layout)
        return tuple(valid)

    def _pointer_occurrences(self, value: int) -> tuple[int, ...]:
        needle = struct.pack("<Q", value)
        found: set[int] = set()
        for address, payload in self._chunks(self._heap_maps()):
            cursor = 0
            while True:
                index = payload.find(needle, cursor)
                if index < 0:
                    break
                location = address + index
                if location % 8 == 0:
                    found.add(location)
                cursor = index + 1
        return tuple(sorted(found))

    def _manager_for_world(self, world: int) -> int | None:
        candidates: list[tuple[int, int]] = []
        for occurrence in self._pointer_occurrences(world):
            manager = occurrence - MANAGER_WORLD
            if manager <= 0 or manager % 8 != 0 or not self._mapped(manager, 0xB0):
                continue
            data = self.memory.read(manager, 0xB0)
            tick = _i32(data, MANAGER_TICK)
            if _u64(data, MANAGER_WORLD) == world and 0 <= tick <= 200_000:
                candidates.append((manager, tick))
        if not candidates:
            return None
        time.sleep(0.12)
        advancing: list[int] = []
        stable: list[int] = []
        for manager, previous_tick in candidates:
            data = self.memory.read(manager, 0xB0)
            tick = _i32(data, MANAGER_TICK)
            if _u64(data, MANAGER_WORLD) != world or not previous_tick <= tick <= previous_tick + 20:
                continue
            (advancing if tick > previous_tick else stable).append(manager)
        if len(advancing) == 1:
            return advancing[0]
        if not advancing and len(stable) == 1:
            return stable[0]
        return None

    def discover(self) -> BattleLayout:
        manager, world = self.discover_manager_world()
        validated = self._validate_world(world)
        if validated is None:
            raise ExternalMemoryError("active GameState world does not match the attested Hog 2.6 layout")
        self.layout = replace(validated, manager=manager)
        self.generation += 1
        self.sequence = 0
        self.events.clear()
        self.event_sequence = 0
        self.seen_objects.clear()
        self.previous_hand = ()
        return self.layout
