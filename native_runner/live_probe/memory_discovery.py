from __future__ import annotations

from dataclasses import replace
import struct
import time
from typing import Sequence

from .public_snapshot import PublicBattleSnapshot
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

    def discover(self) -> BattleLayout:
        manager, world = self.discover_manager_world()
        validated = self._validate_world(world)
        if validated is None:
            raise ExternalMemoryError("active GameState world does not match the attested Hog 2.6 layout")
        self.layout = replace(validated, manager=manager)
        self.generation += 1
        self.public_state = PublicBattleSnapshot(self.card_costs)
        return self.layout
