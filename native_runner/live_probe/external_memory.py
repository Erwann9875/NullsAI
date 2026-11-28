from __future__ import annotations

from typing import Mapping

from .public_snapshot import PublicBattleSnapshot
from .memory_discovery import BattleLayoutDiscoveryMixin
from .memory_snapshot import BattleSnapshotMixin
from .memory_transport import ExternalReaderClient, MemoryTransport
from .memory_types import (
    ARENA_NATIVE_HEIGHT,
    AUTHORIZATION_END_UTC_EXCLUSIVE,
    DECK_SLOT_VECTOR,
    EXPECTED_LIBG_BUILD_ID,
    EXPECTED_LIBG_SHA256,
    HOG_26_DECK,
    HOG_26_SET,
    MANAGER_TICK,
    MANAGER_WORLD,
    MAP_PATTERN,
    MAX_COMPONENTS,
    MAX_EFFECTS,
    MAX_OBJECTS,
    MAX_SNAPSHOT_ATTEMPTS,
    MAX_SNAPSHOT_TICK_SPAN,
    MAX_TRANSFER,
    MAX_WORLD_CANDIDATES,
    PACKAGE,
    ROOT_CYCLE_VECTOR,
    ROOT_ELIXIR,
    ROOT_HAND_VECTOR,
    ROOT_OBJECT_MANAGER,
    SCAN_OVERLAP,
    TOKEN,
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
    NativeVector,
    ProcessMapping,
    _i32,
    _u32,
    _u64,
    _vector,
    parse_maps,
)


class ExternalBattleReader(BattleLayoutDiscoveryMixin, BattleSnapshotMixin):
    def __init__(
        self,
        memory: MemoryTransport,
        *,
        card_costs: Mapping[int, int],
        local_account_id: int,
    ) -> None:
        if isinstance(local_account_id, bool) or int(local_account_id) <= 0:
            raise ValueError("local_account_id must be a positive integer")
        self.memory = memory
        status = memory.status()
        self.pid = int(status["pid"])
        self.maps = memory.maps()
        self.card_costs = {int(card): int(cost) for card, cost in card_costs.items()}
        self.local_account_id = int(local_account_id)
        self.libg_base = self._libg_base()
        self.layout: BattleLayout | None = None
        self.generation = 0
        self.public_state = PublicBattleSnapshot(self.card_costs)

    @property
    def sequence(self) -> int:
        return self.public_state.sequence

    @property
    def event_sequence(self) -> int:
        return self.public_state.event_sequence
