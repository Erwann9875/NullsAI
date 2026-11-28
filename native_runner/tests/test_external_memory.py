from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from native_runner.live_probe.external_memory import (
    ROOT_CYCLE_VECTOR,
    ROOT_ELIXIR,
    ROOT_HAND_VECTOR,
    BattleLayout,
    ExternalBattleReader,
    ProcessMapping,
)


def test_cold_kickoff_class_survives_different_committed_sizes():
    mib = 1024 * 1024
    origin = 0x10000000

    def pool(slot, size):
        start = origin + slot * 256 * mib
        return ProcessMapping(start, start + size, "rw-p", 0, "[anon:scudo:primary]")

    reader = object.__new__(ExternalBattleReader)
    manager = pool(14, 39 * mib)
    world = pool(16, 3 * mib)
    reader.maps = (pool(0, mib), manager, pool(15, mib), world)
    managers, worlds = reader._kickoff_areas()
    assert [(m.start, m.end) for m in managers] == [(manager.end - 8*mib, manager.end)]
    assert worlds == (world,)


def test_cold_kickoff_ignores_unchanged_lobby_records_and_rechecks_changes(monkeypatch):
    reader = object.__new__(ExternalBattleReader)
    reader._lobby_candidates = {16: (80, 1), 32: (96, 3)}
    area = ProcessMapping(0, 16*1024*1024, "rw-p", 0, "[anon:scudo:primary]")
    monkeypatch.setattr(reader, "_candidate_reads", lambda windows: [None]*len(windows))
    monkeypatch.setattr(reader, "_chunk_clock_candidates", lambda chunks, worlds: {16: (80, 1), 32: (96, 4)})
    checked = []
    monkeypatch.setattr(reader, "_moving_candidates", lambda entries: checked.append(entries))
    assert reader._kickoff_clock((area,), (area,), quick=True) is None
    assert checked == [{32: (96, 4)}]


def test_public_frame_uses_fresh_values_and_rechecks_clock_after_a_new_range():
    from native_runner.live_probe.memory_snapshot import PublicFrameReads

    old_object, new_object, clock = (100, 4), (200, 4), (300, 4)
    values = {old_object: 50, new_object: 80, clock: 12}
    calls = []
    def readv(ranges):
        calls.append(tuple(ranges))
        return tuple(struct.pack('<i', values[r]) for r in ranges)
    wire = SimpleNamespace(readv=readv, read=lambda *r: readv([r])[0])
    frame = PublicFrameReads(wire, (old_object,), clock)
    assert frame.readv((old_object,)) == (struct.pack('<i', 50),)
    assert frame.closing_tick() == 12
    assert calls == [(old_object, clock)]
    values[old_object], values[clock] = 40, 13
    another = PublicFrameReads(wire, tuple(frame.used), clock)
    assert another.readv((new_object, old_object)) == (struct.pack('<i', 80), struct.pack('<i', 40))
    values[clock] = 14
    assert another.closing_tick() == 13
    assert calls[-1] == (new_object, clock)


@pytest.mark.parametrize('active', [True,False])
def test_stationary_fast_result_requires_distinct_advancing_local_world(monkeypatch, active):
    from native_runner.live_probe.memory_types import BattleLayout,ExternalMemoryError

    reader=object.__new__(ExternalBattleReader)
    reader.memory=SimpleNamespace(maps=lambda:())
    reader.card_costs={}
    reader.generation=0
    reader.layout=None
    local=BattleLayout(pid=1,libg_base=0,manager=0,world=21,local_identity=2,local_deck=3,local_root=4,
                       opponent_deck=5,opponent_root=6,object_manager=7,account_id=8,deck=(),native_local_owner=0)
    monkeypatch.setattr(reader,'_manager_candidates',lambda:((10,11),))
    monkeypatch.setattr(reader,'_local_world_candidates',lambda:(21,))
    monkeypatch.setattr(reader,'_validate_world',lambda w:local if w==21 else None)
    monkeypatch.setattr(reader,'_manager_for_world',lambda w:20)

    def select(candidates):
        if active and (20,21) in candidates:
            return 20,21
        raise ExternalMemoryError('expected one active GameState manager, found advancing=0 stable=1')

    monkeypatch.setattr(reader,'_select_manager',select)
    if active:
        assert reader.discover().manager==20
        assert reader.layout.world==21
        reader_result = reader.layout
        monkeypatch.setattr(reader, '_moving_local_layout', lambda manager: reader_result)
        monkeypatch.setattr(reader,'_manager_candidates',lambda:pytest.fail('warm attachment ran a scan'))
        assert reader.discover().world==21
    else:
        with pytest.raises(ExternalMemoryError, match='advancing local battle'):
            reader.discover()
        assert reader.layout is None


@pytest.mark.parametrize(
    ("frames", "local", "accepted"),
    [
        ([(100, 60), (102, 60)], True, True),
        ([(100, 60), (100, 60), (100, 60)], True, False),
        ([(100, 60), (99, 60)], True, False),
        ([(100, 60), (102, 64)], True, False),
        ([(100, 60)], False, False),
    ],
)
def test_warm_attach_rechecks_world_owner_and_clock(monkeypatch, frames, local, accepted):
    from native_runner.live_probe import memory_discovery as discovery
    from native_runner.live_probe.memory_types import MANAGER_TICK, MANAGER_WORLD

    frames = iter(frames)

    def read(address, size):
        tick, world = next(frames)
        data = bytearray(size)
        struct.pack_into("<i", data, MANAGER_TICK, tick)
        struct.pack_into("<Q", data, MANAGER_WORLD, world)
        return data

    reader = object.__new__(ExternalBattleReader)
    reader.memory = SimpleNamespace(read=read)
    monkeypatch.setattr(reader, "_mapped", lambda *args, **kwargs: True)
    layout = BattleLayout(1, 2, 0, 60, 3, 4, 5, 6, 7, 8, 9, (), 0)
    monkeypatch.setattr(reader, "_validate_world", lambda world: layout if local else None)
    monkeypatch.setattr(discovery.time, "sleep", lambda seconds: None)
    result = reader._moving_local_layout(16)
    assert (result is not None) == accepted
    if accepted:
        assert (result.manager, result.world) == (16, 60)


class _Memory:
    def __init__(self, payloads: dict[int, bytes]) -> None:
        self.payloads = payloads

    def readv(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        return [self.payloads[address][:size] for address, size in ranges]


@pytest.mark.parametrize(
    ("local_owner", "position", "target", "tower"),
    [
        (0, (5499, 12500), (3500, 6500), (14500, 6500)),
        (1, (12501, 19500), (14500, 25500), (3500, 25500)),
    ],
)
def test_both_seats_map_to_the_same_screen_tiles(local_owner, position, target, tower):
    troop = dict(owner=local_owner, x=position[0], y=position[1], targetX=target[0], targetY=target[1])
    princess = dict(owner=local_owner, x=tower[0], y=tower[1], targetX=tower[0], targetY=tower[1])
    defender, left_tower = ExternalBattleReader._normalize_objects(
        SimpleNamespace(native_local_owner=local_owner), [troop, princess]
    )
    assert (defender["owner"], defender["x"], defender["y"]) == (0, 12501, 12500)
    assert (defender["targetX"], defender["targetY"]) == (14500, 6500)
    assert (left_tower["x"], left_tower["y"]) == (3500, 6500)


@pytest.mark.parametrize('map_state', ['current','stale','unreadable'])
def test_live_hand_and_draw_queue_offsets_match_visible_slots(map_state) -> None:
    from native_runner.live_probe.memory_types import ExternalMemoryError

    deck = (
        26_000_021,
        26_000_014,
        26_000_038,
        27_000_000,
        26_000_010,
        26_000_030,
        28_000_000,
        28_000_011,
    )
    root = bytearray(0x310)
    struct.pack_into("<Qii", root, ROOT_HAND_VECTOR, 0x1000, 4, 4)
    struct.pack_into("<Qii", root, ROOT_CYCLE_VECTOR, 0x2000, 4, 4)
    struct.pack_into("<i", root, ROOT_ELIXIR, 100_000)

    reader = object.__new__(ExternalBattleReader)
    reader.memory = _Memory(
        {

            0x1000: struct.pack("<4i", 3, 5, 4, 6),

            0x2000: struct.pack("<4i", 0, 1, 2, 7),
        }
    )
    reader.maps = (
        ProcessMapping(
            start=0x1000,
            end=0x3000,
            permissions="r--p",
            file_offset=0,
            name="[test]",
        ),
    )
    layout = BattleLayout(
        pid=1,
        libg_base=0,
        manager=0,
        world=0,
        local_identity=0,
        local_deck=0,
        local_root=0,
        opponent_deck=0,
        opponent_root=0,
        object_manager=0,
        account_id=1,
        deck=deck,
        native_local_owner=1,
    )

    if map_state != 'current':
        full_maps = reader.maps
        reader.maps = (ProcessMapping(0x1000,0x1100,'r--p',0,'[test]'),)
        reader.memory.maps = lambda: full_maps if map_state=='stale' else reader.maps
    if map_state == 'unreadable':
        with pytest.raises(ExternalMemoryError, match='cycle vector is not readable'):
            reader._own_state(layout, bytes(root))
        return
    hand, draw_queue, elixir = reader._own_state(layout, bytes(root))

    assert hand == [27_000_000, 26_000_030, 26_000_010, 28_000_000]
    assert draw_queue == [26_000_021, 26_000_014, 26_000_038, 28_000_011]
    assert draw_queue[0] == 26_000_021
    assert elixir == 100_000
