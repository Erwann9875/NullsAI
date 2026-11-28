from __future__ import annotations

from dataclasses import replace
import struct
import time
from bisect import bisect_right
from typing import Sequence

from .public_snapshot import PublicBattleSnapshot
from .world_index import LocalWorldIndex
from .memory_types import (
    DECK_SLOT_VECTOR,
    HOG_26_SET,
    MANAGER_TICK,
    MANAGER_WORLD,
    ROOT_CYCLE_VECTOR,
    ROOT_ELIXIR,
    ROOT_HAND_VECTOR,
    ROOT_OBJECT_MANAGER,
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
    _i32,
    _u64,
    _vector,
)


class BattleLayoutDiscoveryMixin(LocalWorldIndex):
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
        starts = getattr(self, "_map_starts", None)
        if starts is not None:
            index = bisect_right(starts, address) - 1
            if index < 0:
                return False
            mapping = self.maps[index]
            return mapping.readable and (not writable or mapping.writable) and mapping.contains(address, size)
        return any(
            mapping.readable and (not writable or mapping.writable) and mapping.contains(address, size)
            for mapping in self.maps
        )

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
        if data[WORLD_FINALIZED]:
            return None
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

    def _moving_local_layout(self, manager, expected_world=None):
        size = MANAGER_WORLD + 8
        if not self._mapped(manager, size, writable=True):
            return None
        before = self.memory.read(manager, size)
        world = _u64(before, MANAGER_WORLD)
        if expected_world is not None and world != expected_world:
            return None
        first = _i32(before, MANAGER_TICK)
        if not 0 <= first <= 200_000 or not self._mapped(world, 0x200, writable=True):
            return None
        sampled = time.monotonic()
        layout = self._validate_world(world)
        if layout is None:
            return None
        for until in (0.065, 0.16):
            remaining = sampled + until - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            after = self.memory.read(manager, size)
            delta = _i32(after, MANAGER_TICK) - first
            if _u64(after, MANAGER_WORLD) != world or not 0 <= delta <= 50:
                return None
            if delta:
                return replace(layout, manager=manager)
        return None

    def _nearby_heap(self, address, radius):
        for mapping in self.maps:
            if (mapping.name in {"[anon:scudo:primary]", "[anon:scudo:secondary]"}
                    and mapping.readable and mapping.writable and mapping.contains(address)):
                if radius is None:
                    return (mapping,)
                return (replace(mapping, start=max(mapping.start, (address - radius) & ~15),
                                end=min(mapping.end, address + radius)),)
        return ()

    def _cached_layout(self, manager, world, *, quick=False):
        direct = self._moving_local_layout(manager)
        if direct:
            self.discovery_source = "previous manager"
            return direct
        world_area = self._nearby_heap(world, None)
        radii = (768 * 1024,) if quick else (768 * 1024, 3 * 1024 * 1024)
        for radius in radii:
            manager_area = self._nearby_heap(manager, radius)
            if not world_area or not manager_area:
                return None
            moving = self._pool_clock_layout(manager_area, world_area)
            if moving:
                self.discovery_source = "previous battle allocation"
                address, layout = moving
                return replace(layout, manager=address)
        return None

    def _uncached_layout(self, *, quick=False):
        primary = [r for r in self.maps if r.readable and r.writable and r.name == "[anon:scudo:primary]"]
        mib = 1024 * 1024
        managers = [replace(r, start=max(r.start, r.end - 4 * mib)) for r in primary if 35 * mib <= r.size <= 37 * mib]
        worlds = [r for r in primary if 4 * mib <= r.size < 5 * mib]
        if 0 < len(managers) <= 2 and worlds:
            moving = self._pool_clock_layout(managers, worlds)
            if moving:
                self.discovery_source = "game build allocation"
                address, layout = moving
                return replace(layout, manager=address)
        if quick:
            raise ExternalMemoryError("waiting for an advancing local battle")
        candidates = self._manager_candidates()
        try:
            manager, world = self._select_manager(candidates)
            validated = self._validate_world(world)
        except ExternalMemoryError as error:
            if not (str(error).startswith(("expected one active GameState", "no active GameState"))
                    or str(error) == "read range is not contained in a readable mapping"):
                raise
            validated = None
        if validated is None:
            self.discovery_source = "local world index"
            verified = []
            for world in self._local_world_candidates():
                layout = self._validate_world(world)
                if layout is None:
                    continue
                manager = self._manager_for_world(world)
                if manager is None:
                    continue
                try:
                    self._select_manager(((manager, world),))
                except ExternalMemoryError:
                    continue
                verified.append((manager, layout))
            if len(verified) != 1:
                raise ExternalMemoryError(f'expected one advancing local battle, found {len(verified)}')
            manager, validated = verified[0]
        else:
            self.discovery_source = "native game state"
        return replace(validated, manager=manager)

    def discover(self, *, quick=False) -> BattleLayout:
        self.maps = tuple(sorted(self.memory.maps(), key=lambda mapping: mapping.start))
        self._map_starts = tuple(mapping.start for mapping in self.maps)
        self.layout = None
        hint = getattr(self, "_clock_hint", None)
        if hint:
            try:
                self.layout = self._cached_layout(*hint, quick=quick)
            except ExternalMemoryError as error:
                if str(error) != "read range is not contained in a readable mapping":
                    raise
        if self.layout is None:
            self.layout = self._uncached_layout(quick=quick)
        self._clock_hint = (self.layout.manager, self.layout.world)
        self.generation += 1
        self.public_state = PublicBattleSnapshot(self.card_costs)
        return self.layout
