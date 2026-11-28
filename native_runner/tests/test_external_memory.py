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


def test_live_hand_and_draw_queue_offsets_match_visible_slots() -> None:

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

    hand, draw_queue, elixir = reader._own_state(layout, bytes(root))

    assert hand == [27_000_000, 26_000_030, 26_000_010, 28_000_000]
    assert draw_queue == [26_000_021, 26_000_014, 26_000_038, 28_000_011]
    assert draw_queue[0] == 26_000_021
    assert elixir == 100_000
