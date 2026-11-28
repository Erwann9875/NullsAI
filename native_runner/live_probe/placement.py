from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..arena import OccupiedFootprintV1, PlacementMaskV1, card_placement_mask, native_building_footprint
from ..battle_env import _earliest_affordable_bin, _touch_window_elixir_raw
from ..contracts import ActionKind, ActionMaskV1, CardSpecV1, EntityStateV1, FrozenMapping, TowerStateV1
from .live_contracts import LiveLadderError
from .live_settings import RULESET_ID
from .play_ledger import PendingTouch


class PlacementMaskBuilder:
    def __init__(self, card_specs: Mapping[int, CardSpecV1]) -> None:
        self.card_specs = card_specs

    @staticmethod
    def _tower_footprints(towers: Sequence[TowerStateV1]) -> tuple[OccupiedFootprintV1, ...]:
        return tuple(
            OccupiedFootprintV1(
                float(tower.position[0]),
                float(tower.position[1]),
                4 if tower.tower_kind == "king" else 3,
                4 if tower.tower_kind == "king" else 3,
                f"tower:{tower.entity_id}",
            )
            for tower in towers
            if tower.active
        )

    def _building_footprints(self, entities: Sequence[EntityStateV1]) -> tuple[OccupiedFootprintV1, ...]:
        result: list[OccupiedFootprintV1] = []
        for entity in entities:
            if entity.card_id is None or entity.hitpoints == 0:
                continue
            spec = self.card_specs.get(entity.card_id)
            if spec is None or spec.kind.value != "building":
                continue
            footprint = native_building_footprint(spec)
            if footprint is None:
                raise LiveLadderError(f"live building {entity.card_id} has no exact deployment footprint")
            result.append(
                OccupiedFootprintV1(
                    float(entity.position[0]),
                    float(entity.position[1]),
                    footprint.width_tiles,
                    footprint.height_tiles,
                    f"building:{entity.entity_id}",
                )
            )
        return tuple(result)

    @staticmethod
    def _placement(slot: int, card_id: int, cost: float, artifact: PlacementMaskV1) -> FrozenMapping:
        return FrozenMapping(
            {
                "card_id": card_id,
                "visible_card_id": card_id,
                "effective_card_id": card_id,
                "native_effective_card_id": card_id,
                "effective_cost": cost,
                "form_code": 0,
                "native_form_code": 0,
                "source_native_hand_slot": slot,
                "shape": (32, 18),
                "row_major": artifact.rows,
                "placement_rule": artifact.rule.value,
                "accuracy": artifact.accuracy.value,
                "mask_id": artifact.mask_id,
                "algorithm": artifact.algorithm,
                "footprint_width_tiles": artifact.footprint_width_tiles,
                "footprint_height_tiles": artifact.footprint_height_tiles,
                "model_subcell_offset": artifact.model_subcell_offset,
                "reasons": artifact.reasons,
            }
        )

    def build(
        self,
        tick: int,
        own: Mapping[str, object],
        pending: Sequence[PendingTouch],
        towers: Sequence[TowerStateV1],
        entities: Sequence[EntityStateV1],
    ) -> ActionMaskV1:
        elixir_raw = int(own["elixirRaw"])
        reserved_raw = int(round(sum(item.cost for item in pending if not item.slot_changed) * 10_000))
        effective_raw = max(0, elixir_raw - reserved_raw)
        window = _touch_window_elixir_raw(tick, effective_raw)
        active_towers = self._tower_footprints(towers)
        occupied = self._building_footprints(entities)
        destroyed_enemy_lanes = tuple(
            "left" if tower.tower_kind.endswith("left") else "right"
            for tower in towers
            if tower.owner == 1 and tower.tower_kind.startswith("princess_") and not tower.active
        )
        hand = tuple(int(card) for card in own["hand"])
        slots: list[bool] = []
        placements: dict[str, object] = {}
        reasons: dict[str, object] = {}
        affordable_from: dict[str, int] = {}
        reserved_cards = {item.action.card_id for item in pending}
        for slot, card_id in enumerate(hand):
            spec = self.card_specs[card_id]
            if spec.elixir_cost is None:
                raise LiveLadderError(f"Hog card {card_id} has no exact runtime cost")
            artifact = card_placement_mask(
                spec,
                0,
                ruleset_id=RULESET_ID,
                destroyed_enemy_princess_lanes=destroyed_enemy_lanes,
                active_tower_footprints=active_towers,
                occupied_building_footprints=occupied,
            )
            cost = float(spec.elixir_cost)
            first_bin = _earliest_affordable_bin(round(cost * 10_000), window)
            if card_id in reserved_cards:
                reason = "pending"
            elif first_bin is None or not any(any(row) for row in artifact.rows):
                reason = "insufficient_elixir"
            else:
                reason = "legal"
            slots.append(reason == "legal")
            reasons[str(slot)] = reason
            if first_bin is not None:
                affordable_from[str(slot)] = first_bin
            placements[str(slot)] = self._placement(slot, card_id, cost, artifact)
        return ActionMaskV1(
            kinds={
                ActionKind.WAIT.value: True,
                ActionKind.WAIT_UNTIL_AFFORDABLE.value: any(not value for value in slots),
                ActionKind.PLAY_CARD.value: any(slots),
                ActionKind.ACTIVATE_ABILITY.value: False,
            },
            hand_slots=tuple(slots),
            placement_masks=placements,
            reasons={
                **reasons,
                "raw_elixir": elixir_raw / 10_000.0,
                "reserved_elixir": reserved_raw / 10_000.0,
                "effective_elixir": effective_raw / 10_000.0,
                "affordable_from_bin": affordable_from,
                "decision_ready": True,
            },
        )
