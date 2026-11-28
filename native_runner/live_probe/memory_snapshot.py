from __future__ import annotations

import struct
import time
from typing import Mapping, Sequence

from .memory_types import (
    ARENA_NATIVE_HEIGHT,
    ARENA_NATIVE_WIDTH,
    MANAGER_TICK,
    MANAGER_WORLD,
    MAX_COMPONENTS,
    MAX_EFFECTS,
    MAX_OBJECTS,
    MAX_SNAPSHOT_ATTEMPTS,
    MAX_SNAPSHOT_TICK_SPAN,
    ROOT_CYCLE_VECTOR,
    ROOT_ELIXIR,
    ROOT_HAND_VECTOR,
    ROOT_OBJECT_MANAGER,
    WORLD_FINALIZED,
    WORLD_LOCAL_DECK,
    WORLD_LOCAL_IDENTITY,
    WORLD_LOCAL_ROOT,
    WORLD_RESULT,
    BattleLayout,
    ExternalMemoryError,
    _i32,
    _u32,
    _u64,
    _vector,
)


class BattleSnapshotMixin:
    def _own_state(self, layout: BattleLayout, root: bytes) -> tuple[list[int], list[int], int]:
        hand = _vector(root, ROOT_HAND_VECTOR, 16)
        cycle = _vector(root, ROOT_CYCLE_VECTOR, 32)
        if hand is None or cycle is None or hand.count > 4 or cycle.count > 8:
            raise ExternalMemoryError("local hand/cycle vectors failed validation")
        ranges: list[tuple[int, int]] = []
        labels: list[str] = []
        for name, vector in (("hand", hand), ("cycle", cycle)):
            if vector.count:
                if not self._mapped(vector.data, vector.count * 4):
                    raise ExternalMemoryError(f"local {name} vector is not readable")
                ranges.append((vector.data, vector.count * 4))
                labels.append(name)
        payloads = self.memory.readv(ranges)
        raw: dict[str, tuple[int, ...]] = {"hand": (), "cycle": ()}
        for name, payload in zip(labels, payloads, strict=True):
            raw[name] = struct.unpack(f"<{len(payload) // 4}i", payload)
        output: dict[str, list[int]] = {"hand": [], "cycle": []}
        for name in ("hand", "cycle"):
            for slot in raw[name]:
                if name == "hand" and slot < 0:
                    continue
                if not 0 <= slot < 8:
                    raise ExternalMemoryError(f"local {name} contains invalid deck slot {slot}")
                output[name].append(layout.deck[slot])
        if len(output["hand"]) != 4 or len(set(output["hand"])) != 4:
            raise ExternalMemoryError("local live hand is incomplete")
        elixir = _i32(root, ROOT_ELIXIR)
        if not 0 <= elixir <= 100_000:
            raise ExternalMemoryError("local elixir is outside its exact native range")
        return output["hand"], output["cycle"], elixir

    def _objects(self, layout: BattleLayout, root: bytes) -> list[dict[str, object]]:
        object_manager = _u64(root, ROOT_OBJECT_MANAGER)
        if object_manager != layout.object_manager:
            raise ExternalMemoryError("native object manager changed during the battle")
        manager_data = self.memory.read(object_manager, 0x18)
        vector = _vector(manager_data, 0x08, 100_000)
        if vector is None or vector.count > MAX_OBJECTS or not self._mapped(vector.data, vector.count * 8):
            raise ExternalMemoryError("public object vector is invalid or exceeds its fair snapshot bound")
        pointers = struct.unpack(f"<{vector.count}Q", self.memory.read(vector.data, vector.count * 8))
        object_pairs = [(pointer, 0xB0) for pointer in pointers if pointer and self._mapped(pointer, 0xB0)]
        object_data = self.memory.readv(object_pairs)
        bases = {pointer: data for (pointer, _), data in zip(object_pairs, object_data, strict=True)}

        component_ranges: dict[int, tuple[int, int]] = {}
        logic_data_pointers: set[int] = set()
        for pointer, data in bases.items():
            components = _u64(data, 0x18)
            capacity = _i32(data, 0x20)
            count = _i32(data, 0x24)
            if 0 <= count <= capacity <= MAX_COMPONENTS and (count == 0 or self._mapped(components, count * 8)):
                if count:
                    component_ranges[pointer] = (components, count * 8)
            logic = _u64(data, 0x48)
            if logic and self._mapped(logic, 0x44):
                logic_data_pointers.add(logic)

        component_payloads = self.memory.readv(tuple(component_ranges.values()))
        components_by_object = {
            pointer: struct.unpack(f"<{len(payload) // 8}Q", payload)
            for pointer, payload in zip(component_ranges, component_payloads, strict=True)
        }
        component_pointers = {
            component
            for components in components_by_object.values()
            for component in components[2:4]
            if component and self._mapped(component, 0x38)
        }
        all_ranges = tuple((pointer, 0x44) for pointer in sorted(logic_data_pointers)) + tuple(
            (pointer, 0x38) for pointer in sorted(component_pointers)
        )
        all_payloads = self.memory.readv(all_ranges)
        payload_by_pointer = {
            pointer: payload for (pointer, _), payload in zip(all_ranges, all_payloads, strict=True)
        }

        result: list[dict[str, object]] = []
        for pointer, data in bases.items():
            native_id = _u32(data, 0x08)
            owner = _i32(data, 0x78)
            if native_id == 0 or owner < -1 or owner > 1:
                continue
            components = components_by_object.get(pointer, ())
            if owner in (0, 1) and owner != layout.native_local_owner and len(components) > 3 and components[3] != 0:
                visibility = payload_by_pointer.get(components[3])
                if visibility is None or _u64(visibility, 0x08) != pointer:
                    continue
                effect_capacity = _i32(visibility, 0x20)
                effect_count = _i32(visibility, 0x24)
                invisible_count = _i32(visibility, 0x34)
                if (
                    effect_count < 0
                    or effect_capacity < effect_count
                    or effect_capacity > MAX_EFFECTS
                    or invisible_count < 0
                    or invisible_count > effect_count
                    or invisible_count > 0
                ):
                    continue
            logic_pointer = _u64(data, 0x48)
            logic = payload_by_pointer.get(logic_pointer)
            if logic is None:
                continue
            global_id = _u32(logic, 0x40)
            if global_id == 0:
                continue
            hp: int | None = None
            maximum: int | None = None
            if len(components) > 2 and components[2] != 0:
                hitpoints = payload_by_pointer.get(components[2])
                if hitpoints is not None and _u64(hitpoints, 0x08) == pointer:
                    candidate_hp = _i32(hitpoints, 0x10)
                    candidate_maximum = _i32(hitpoints, 0x14)
                    if 0 <= candidate_hp <= candidate_maximum and candidate_maximum > 0:
                        hp = candidate_hp
                        maximum = candidate_maximum
            result.append(
                {
                    "nativeObjectId": native_id,
                    "owner": owner,
                    "cardId": _i32(data, 0xAC),
                    "dataGlobalId": global_id,
                    "x": _i32(data, 0x7C),
                    "y": _i32(data, 0x80),
                    "targetX": _i32(data, 0x84),
                    "targetY": _i32(data, 0x88),
                    "hp": hp,
                    "maxHp": maximum,
                }
            )
        return result

    @staticmethod
    def _normalize_objects(layout: BattleLayout, objects: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        # In seat zero, increasing native x runs toward screen left.
        if layout.native_local_owner == 0:
            return [
                dict(item, x=ARENA_NATIVE_WIDTH - int(item["x"]), targetX=ARENA_NATIVE_WIDTH - int(item["targetX"]))
                for item in objects
            ]
        normalized: list[dict[str, object]] = []
        for item in objects:
            value = dict(item)
            owner = int(value["owner"])
            if owner in (0, 1):
                value["owner"] = 1 - owner
            value["y"] = ARENA_NATIVE_HEIGHT - int(value["y"])
            value["targetY"] = ARENA_NATIVE_HEIGHT - int(value["targetY"])
            normalized.append(value)
        return normalized

    def snapshot(self, after_event: int = 0) -> dict[str, object]:
        layout = self.layout or self.discover()
        for _attempt in range(MAX_SNAPSHOT_ATTEMPTS):
            manager, world, root, identity = self.memory.readv(
                (
                    (layout.manager, 0xB0),
                    (layout.world, 0x200),
                    (layout.local_root, 0x310),
                    (layout.local_identity, 8),
                )
            )
            if _u64(manager, MANAGER_WORLD) != layout.world:
                raise ExternalMemoryError("active battle manager changed; rediscovery is required")
            owner_offset = layout.native_local_owner * 8
            if (
                _u64(world, WORLD_LOCAL_ROOT + owner_offset) != layout.local_root
                or _u64(world, WORLD_LOCAL_DECK + owner_offset) != layout.local_deck
                or _u64(world, WORLD_LOCAL_IDENTITY + owner_offset) != layout.local_identity
            ):
                raise ExternalMemoryError("active battle world changed; rediscovery is required")
            if self._account_id(identity) != layout.account_id:
                raise ExternalMemoryError("local account identity changed")
            tick_before = _i32(manager, MANAGER_TICK)
            hand, cycle, elixir = self._own_state(layout, root)
            objects = self._normalize_objects(layout, self._objects(layout, root))
            tick_after = _i32(self.memory.read(layout.manager + MANAGER_TICK, 4), 0)
            if tick_before <= tick_after <= tick_before + MAX_SNAPSHOT_TICK_SPAN:
                break
        else:
            raise ExternalMemoryError("native state changed across three bounded snapshot attempts")
        return self.public_state.publish(
            generation=self.generation,
            tick=tick_after,
            finalized=bool(world[WORLD_FINALIZED]),
            world_result=_i32(world, WORLD_RESULT),
            account_id=layout.account_id,
            deck=layout.deck,
            hand=hand,
            cycle=cycle,
            elixir=elixir,
            objects=objects,
            after_event=after_event,
        )
