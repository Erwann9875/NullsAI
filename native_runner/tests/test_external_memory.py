from __future__ import annotations

import struct

from native_runner.live_probe.external_memory import (
    ROOT_CYCLE_VECTOR,
    ROOT_ELIXIR,
    ROOT_HAND_VECTOR,
    BattleLayout,
    ExternalBattleReader,
    ProcessMapping,
)


class _Memory:
    def __init__(self, payloads: dict[int, bytes]) -> None:
        self.payloads = payloads

    def readv(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        return [self.payloads[address][:size] for address, size in ranges]


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
