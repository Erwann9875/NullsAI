from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

from .game_profile import AUTHORIZATION_END_EPOCH_EXCLUSIVE, EXPECTED_SCHEMA
from .memory_types import ExternalMemoryError


class PublicBattleSnapshot:
    def __init__(self, card_costs: Mapping[int, int]) -> None:
        self.card_costs = card_costs
        self.sequence = 0
        self.event_sequence = 0
        self.events: deque[dict[str, object]] = deque(maxlen=256)
        self.seen_objects: set[int] = set()
        self.previous_hand: tuple[int, ...] = ()
        self.previous_tower_troops = [0, 0]

    @staticmethod
    def _crowns(objects: Sequence[Mapping[str, object]]) -> list[int]:
        alive = {(owner, kind): False for owner in (0, 1) for kind in ("king", "left", "right")}
        positions = {
            (0, 9_000, 3_000): "king",
            (0, 3_500, 6_500): "left",
            (0, 14_500, 6_500): "right",
            (1, 9_000, 29_000): "king",
            (1, 3_500, 25_500): "left",
            (1, 14_500, 25_500): "right",
        }
        for item in objects:
            key = (int(item["owner"]), int(item["x"]), int(item["y"]))
            kind = positions.get(key)
            if kind is not None and int(item["cardId"]) == -1 and item["hp"] is not None:
                alive[(key[0], kind)] = int(item["hp"]) > 0
        crowns = [0, 0]
        for owner in (0, 1):
            opponent = 1 - owner
            crowns[owner] = (
                3
                if not alive[(opponent, "king")]
                else int(not alive[(opponent, "left")]) + int(not alive[(opponent, "right")])
            )
        return crowns

    def _tower_troops(self, objects: Sequence[Mapping[str, object]]) -> list[int]:
        lookup = {
            35_000_001: 159_000_000,
            1_485_117_103: 159_000_001,
            2_545_895_475: 159_000_002,
            3_191_249_505: 159_000_004,
        }
        output = [0, 0]
        for owner in (0, 1):
            candidates: set[int] = set()
            chef_king = False
            for item in objects:
                if int(item["owner"]) != owner or int(item["cardId"]) != -1:
                    continue
                x, y, global_id = int(item["x"]), int(item["y"]), int(item["dataGlobalId"])
                if x == 9_000 and y == (3_000 if owner == 0 else 29_000) and global_id == 1_471_932_914:
                    chef_king = True
                if y == (6_500 if owner == 0 else 25_500) and x in {3_500, 14_500} and global_id in lookup:
                    candidates.add(lookup[global_id])
            if chef_king:
                output[owner] = 159_000_004
            elif len(candidates) == 1:
                output[owner] = candidates.pop()
            else:
                output[owner] = self.previous_tower_troops[owner]
            if output[owner] == 0:
                raise ExternalMemoryError("public towers do not identify both Tower Troops")
        self.previous_tower_troops = output.copy()
        return output

    def _publish_card_events(
        self, tick: int, hand: tuple[int, ...], objects: Sequence[Mapping[str, object]]
    ) -> None:
        played: set[tuple[int, int]] = set()
        if self.previous_hand:
            for card in set(self.previous_hand).difference(hand):
                played.add((0, card))
        for item in objects:
            native_id = int(item["nativeObjectId"])
            if native_id in self.seen_objects:
                continue
            self.seen_objects.add(native_id)
            owner = int(item["owner"])
            card = int(item["cardId"])
            if owner in (0, 1) and card > 0:
                played.add((owner, card))
        self.previous_hand = hand
        for owner, card in sorted(played):
            cost = self.card_costs.get(card)
            if cost is None:
                continue
            self.event_sequence += 1
            self.events.append(
                {
                    "sequence": self.event_sequence,
                    "tick": tick,
                    "kind": "card_play",
                    "owner": owner,
                    "cardId": card,
                    "cost": cost,
                    "formCode": 0,
                }
            )

    def publish(
        self,
        *,
        generation: int,
        tick: int,
        finalized: bool,
        world_result: int,
        account_id: int,
        deck: Sequence[int],
        hand: list[int],
        cycle: list[int],
        elixir: int,
        objects: list[dict[str, object]],
        after_event: int,
    ) -> dict[str, object]:
        crowns = self._crowns(objects)
        tower_troops = self._tower_troops(objects)
        self._publish_card_events(tick, tuple(hand), objects)
        first_retained = self.events[0]["sequence"] if self.events else self.event_sequence + 1
        events_truncated = after_event + 1 < int(first_retained)
        events = [event for event in self.events if int(event["sequence"]) > after_event]
        self.sequence += 1
        return {
            "ok": True,
            "schema": EXPECTED_SCHEMA,
            "generation": generation,
            "sequence": self.sequence,
            "tick": tick,
            "ended": finalized,
            "finalized": finalized,
            "worldResultRaw": world_result,
            "localOwner": 0,
            "authorizationEndUtcExclusive": AUTHORIZATION_END_EPOCH_EXCLUSIVE,
            "players": [
                {
                    "owner": 0,
                    "accountId": account_id,
                    "crownsRaw": crowns[0],
                    "elixirRaw": elixir,
                    "elixir": elixir / 10_000.0,
                    "towerTroopId": tower_troops[0],
                    "privateStateVisible": True,
                    "hand": hand,
                    "cycle": cycle,
                    "nextCard": cycle[0] if cycle else None,
                    "deck": list(deck),
                },
                {
                    "owner": 1,
                    "crownsRaw": crowns[1],
                    "towerTroopId": tower_troops[1],
                    "elixirRaw": None,
                    "elixir": None,
                    "privateStateVisible": False,
                    "hand": [],
                    "cycle": [],
                    "nextCard": None,
                    "deck": [],
                },
            ],
            "objects": objects,
            "returned": len(objects),
            "objectsTruncated": False,
            "eventsTruncated": events_truncated,
            "events": events,
            "eventCursor": self.event_sequence,
        }
