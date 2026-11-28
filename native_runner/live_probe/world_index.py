from __future__ import annotations

import struct
import time
from typing import Iterable

import numpy as np

from .memory_types import (
    ExternalMemoryError, MANAGER_TICK, MANAGER_WORLD, MAX_TRANSFER,
    MAX_WORLD_CANDIDATES, ProcessMapping, SCAN_OVERLAP, WORLD_LOCAL_IDENTITY,
    WORLD_OPPONENT_IDENTITY, _i32, _u64,
)


class LocalWorldIndex:
    def _pool_candidates(self, manager_areas, world_areas):
        return self._chunk_clock_candidates(self._chunks(manager_areas), world_areas)

    def _chunk_clock_candidates(self, chunks, world_areas):
        candidates = {}
        for base, data in chunks:
            start = (-base) % 16
            n = (len(data) - start - MANAGER_WORLD - 8) // 16 + 1
            if n <= 0:
                continue
            worlds = np.ndarray(n, "<u8", buffer=data, offset=start + MANAGER_WORLD, strides=(16,))
            ticks = np.ndarray(n, "<i4", buffer=data, offset=start + MANAGER_TICK, strides=(16,))
            eligible = np.zeros(n, dtype=bool)
            for region in world_areas:
                eligible |= (worlds >= region.start) & (worlds <= region.end - 0x200)
            eligible &= ((worlds & 15) == 0) & (ticks >= 0) & (ticks <= 200_000)
            for i in np.flatnonzero(eligible):
                candidates[base + start + 16 * int(i)] = (int(worlds[i]), int(ticks[i]))
            if len(candidates) > 1024:
                return {}
        return candidates

    def _pool_clock_layout(self, manager_areas, world_areas):
        return self._moving_candidates(self._pool_candidates(manager_areas, world_areas))

    def _moving_candidates(self, candidates):
        if not candidates:
            return None
        size = MANAGER_WORLD + 8 - MANAGER_TICK
        addresses = tuple(candidates)
        ranges = tuple((address + MANAGER_TICK, size) for address in addresses)
        sampled = time.monotonic()
        for delay in (0.065, 0.15):
            time.sleep(max(0, sampled + delay - time.monotonic()))
            rows = self._candidate_reads(ranges)
            valid = []
            for address, row in zip(addresses, rows, strict=True):
                world, tick = candidates[address]
                if row is None or _u64(row, MANAGER_WORLD - MANAGER_TICK) != world:
                    continue
                if 0 < _i32(row, 0) - tick <= 50:
                    layout = self._validate_world(world)
                    if layout is not None:
                        valid.append((address, layout))
            if len(valid) == 1:
                return valid[0]
            if valid:
                break
        return None

    def _candidate_reads(self, ranges):
        if not ranges:
            return ()
        try:
            return self.memory.readv(ranges)
        except ExternalMemoryError as error:
            if str(error) != "read range is not contained in a readable mapping":
                raise
            if len(ranges) == 1:
                return (None,)
            mid = len(ranges) // 2
            return (*self._candidate_reads(ranges[:mid]), *self._candidate_reads(ranges[mid:]))

    def _local_world_candidates(self, mappings=None) -> tuple[int, ...]:
        worlds = self._world_candidates(mappings)
        selected = []
        for start in range(0, len(worlds), 128):
            batch = worlds[start:start + 128]
            span = WORLD_OPPONENT_IDENTITY - WORLD_LOCAL_IDENTITY + 8
            headers = self._candidate_reads(tuple((w + WORLD_LOCAL_IDENTITY, span) for w in batch))
            owners = [(_u64(h, 0), _u64(h, span - 8)) if h is not None else (0, 0) for h in headers]
            readable = tuple(dict.fromkeys(p for pair in owners for p in pair if self._mapped(p, 8)))
            ids = self._candidate_reads(tuple((p, 8) for p in readable))
            local = {p for p, identity in zip(readable, ids, strict=True)
                     if identity is not None and self._account_id(identity) == self.local_account_id}
            selected.extend(w for w, pair in zip(batch, owners, strict=True)
                            if sum(p in local for p in pair) == 1)
        return tuple(selected)

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

    def _world_candidates(self, mappings=None) -> tuple[int, ...]:
        heap = self._heap_maps()
        heap_starts = np.asarray([mapping.start for mapping in heap], dtype=np.uint64)
        heap_ends = np.asarray([mapping.end for mapping in heap], dtype=np.uint64)
        found: set[int] = set()
        primary = tuple(mapping for mapping in heap if mapping.name.endswith(":primary]"))
        phases = (primary, tuple(mapping for mapping in heap if mapping not in primary)) if mappings is None else (mappings,)
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

    def _pointer_occurrences(self, value: int, mappings=None) -> tuple[int, ...]:
        needle = struct.pack("<Q", value)
        found: set[int] = set()
        for address, payload in self._chunks(self._heap_maps() if mappings is None else mappings):
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
            (data,) = self._candidate_reads(((manager, 0xB0),))
            if data is None:
                continue
            tick = _i32(data, MANAGER_TICK)
            if _u64(data, MANAGER_WORLD) == world and 0 <= tick <= 200_000:
                candidates.append((manager, tick))
        if not candidates:
            return None
        time.sleep(0.12)
        advancing: list[int] = []
        stable: list[int] = []
        for manager, previous_tick in candidates:
            (data,) = self._candidate_reads(((manager, 0xB0),))
            if data is None:
                continue
            tick = _i32(data, MANAGER_TICK)
            if _u64(data, MANAGER_WORLD) != world or not previous_tick <= tick <= previous_tick + 20:
                continue
            (advancing if tick > previous_tick else stable).append(manager)
        if len(advancing) == 1:
            return advancing[0]
        if not advancing and len(stable) == 1:
            return stable[0]
        return None
