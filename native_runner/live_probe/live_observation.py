from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import time
from typing import Mapping, Sequence

from ..arena import OccupiedFootprintV1, card_placement_mask, native_building_footprint
from ..battle_env import (
    TOWER_LAYOUT,
    _earliest_affordable_bin,
    _fair_snapshot_damage_contract,
    _touch_window_elixir_raw,
)
from ..contracts import (
    ActionKind,
    ActionMaskV1,
    ActionV1,
    EntityStateV1,
    EventV1,
    FrozenMapping,
    ObservationTier,
    ObservationV1,
    PendingActionV1,
    PlayerStateV1,
    TerminalV1,
    TimeStateV1,
    TowerStateV1,
)
from ..cr_native_env import COMMAND_EXECUTION_DELAY_TICKS
from ..match_factory import NATIVE_GAMEPLAY_END_TICK
from ..training.card_features import HERO_FORM_TO_BASE_CARD
from ..training.v4.catalog import UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
from ..training.v4.factory import production_semantic_bundle
from .live_contracts import LiveLadderError, validate_fair_payload
from .live_settings import (
    EPISODE_PREFIX,
    EXPECTED_SCHEMA,
    RECEIPT_TIMEOUT_TICKS,
    RULESET_ID,
    TICK_MS,
    TOWER_POSITION,
)


class PendingTouch:
    action: ActionV1
    cost: float
    requested_tick: int
    expected_tick: int
    source_cursor: int
    slot_changed: bool = False
    event_seen: bool = False

class LiveObservationBuilder:

    def __init__(self) -> None:
        self.bundle = production_semantic_bundle()
        self.generation: int | None = None
        self.episode_id: str | None = None
        self.first_seen: dict[int, int] = {}
        self.previous_entities: dict[int, EntityStateV1] = {}
        self.previous_towers: dict[int, TowerStateV1] = {}
        self.tower_registry: dict[tuple[int, str], tuple[int, tuple[int, int], float]] = {}
        self.history: deque[EventV1] = deque()
        self.seen_public_events: set[int] = set()
        self.opponent_revealed: set[int] = set()
        self.pending: list[PendingTouch] = []
        self.receipts: list[tuple[PendingTouch, str]] = []
        self.last_tick = 0

    def _reset(self, generation: int) -> None:
        self.generation = generation
        self.episode_id = f"{EPISODE_PREFIX}-{generation}"
        self.first_seen.clear()
        self.previous_entities.clear()
        self.previous_towers.clear()
        self.tower_registry.clear()
        self.history.clear()
        self.seen_public_events.clear()
        self.opponent_revealed.clear()
        self.pending.clear()
        self.receipts.clear()
        self.last_tick = 0

    @staticmethod
    def _phase(tick: int) -> tuple[str, float]:
        multiplier = 1.0 if tick < 2400 else 2.0 if tick < 4800 else 3.0
        phase = "regulation" if tick < 3600 else "overtime" if tick < NATIVE_GAMEPLAY_END_TICK else "tiebreak"
        return phase, multiplier

    @staticmethod
    def _tower_troop_for_kind(configured: int, kind: str) -> int | None:
        if configured == 159_000_004:
            return configured if kind == "king" else 159_000_000
        return None if kind == "king" else configured

    def _towers(self, raw: Mapping[str, object]) -> tuple[TowerStateV1, ...]:
        players = raw["players"]
        assert isinstance(players, list)
        troop_by_owner = {int(player["owner"]): int(player["towerTroopId"]) for player in players}
        current: dict[tuple[int, str], Mapping[str, object]] = {}
        for item in raw["objects"]:
            assert isinstance(item, Mapping)
            key = TOWER_POSITION.get((int(item["owner"]), int(item["x"]), int(item["y"])))
            if int(item["cardId"]) != -1 or key is None:
                continue
            entity_id, kind = key
            hp = item.get("hp")
            maximum = item.get("maxHp")
            if hp is None or maximum is None:
                raise LiveLadderError("a public tower has no exact HP")
            current[(int(item["owner"]), kind)] = item
            self.tower_registry[(int(item["owner"]), kind)] = (
                int(item["nativeObjectId"]),
                (int(item["x"]), int(item["y"])),
                float(maximum),
            )
        if not self.tower_registry and len(current) != 6:
            raise LiveLadderError("the first live frame does not contain all six towers")
        towers: list[TowerStateV1] = []
        for _fixed_id, owner, kind, x, y in TOWER_LAYOUT:
            key = (owner, kind)
            item = current.get(key)
            registry = self.tower_registry.get(key)
            if registry is None:
                raise LiveLadderError(f"tower identity {key} was never observed")
            entity_id, position, maximum = registry
            hp = 0.0 if item is None else float(item["hp"])
            target = None
            towers.append(
                TowerStateV1(
                    entity_id=entity_id,
                    owner=owner,
                    tower_kind=kind,
                    position=position,
                    hitpoints=hp,
                    max_hitpoints=maximum,
                    tower_troop_id=self._tower_troop_for_kind(troop_by_owner[owner], kind),
                    active=hp > 0,
                    visible_target=target,
                )
            )
        return tuple(towers)

    def _entity_kind(self, data_global_id: int, card_id: int | None) -> str:
        catalog = self.bundle.entity_archetype_catalog
        vocab = catalog.runtime_global_vocab_id(data_global_id)
        if vocab == UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID:
            raise LiveLadderError(f"public entity dataGlobalId {data_global_id} is outside the checkpoint vocabulary")
        child_kind = catalog.metadata_for_vocab_id(vocab).child_kind
        if child_kind == "character":
            if card_id is not None:
                spec = self.bundle.card_specs.get(card_id)
                if spec is not None and spec.kind.value == "building":
                    return "building"
            return "troop"
        if child_kind not in {"building", "projectile", "area", "effect"}:
            raise LiveLadderError(f"public entity {data_global_id} has unsupported child kind {child_kind!r}")
        return child_kind

    def _entities(self, raw: Mapping[str, object], tick: int) -> tuple[EntityStateV1, ...]:
        result: list[EntityStateV1] = []
        dt = tick - self.last_tick
        for item in raw["objects"]:
            assert isinstance(item, Mapping)
            if int(item["cardId"]) == -1 and TOWER_POSITION.get(
                (int(item["owner"]), int(item["x"]), int(item["y"]))
            ) is not None:
                continue
            entity_id = int(item["nativeObjectId"])
            raw_card_id = int(item["cardId"])
            card_id = HERO_FORM_TO_BASE_CARD.get(raw_card_id, raw_card_id)
            if card_id not in self.bundle.card_specs:
                card_id = None
            data_global_id = int(item["dataGlobalId"])
            position = (float(item["x"]), float(item["y"]))
            previous = self.previous_entities.get(entity_id)
            velocity = None
            if previous is not None and dt > 0:
                velocity = ((position[0] - previous.position[0]) / dt, (position[1] - previous.position[1]) / dt)
            target_x, target_y = int(item["targetX"]), int(item["targetY"])
            movement_target = (
                (float(target_x), float(target_y))
                if 0 <= target_x < 18_000 and 0 <= target_y < 32_000 and (target_x, target_y) != position
                else None
            )
            hp = item.get("hp")
            maximum = item.get("maxHp")
            result.append(
                EntityStateV1(
                    entity_id=entity_id,
                    owner=(int(item["owner"]) if int(item["owner"]) in (0, 1) else None),
                    card_id=card_id,
                    entity_kind=self._entity_kind(data_global_id, card_id),
                    native_data_global_id=data_global_id,
                    position=position,
                    velocity=velocity,
                    hitpoints=None if hp is None else float(hp),
                    max_hitpoints=None if maximum is None else float(maximum),
                    age_ms=(tick - self.first_seen.setdefault(entity_id, tick)) * TICK_MS,
                    movement_target=movement_target,
                    visible=True,
                )
            )
        return tuple(result)

    def _append_probe_events(self, raw: Mapping[str, object]) -> None:
        for item in raw["events"]:
            assert isinstance(item, Mapping)
            sequence = int(item["sequence"])
            if sequence in self.seen_public_events:
                continue
            self.seen_public_events.add(sequence)
            card_id = int(item["cardId"])
            if card_id not in self.bundle.card_specs:
                raise LiveLadderError(f"publicly played card {card_id} is outside the checkpoint vocabulary")
            owner = int(item["owner"])
            if owner == 1:
                self.opponent_revealed.add(card_id)
                if len(self.opponent_revealed) > 8:
                    raise LiveLadderError("opponent publicly revealed more than eight cards")
            cost = float(item["cost"])
            static_cost = self.bundle.card_specs[card_id].elixir_cost
            if static_cost is None:
                raise LiveLadderError(f"public card {card_id} has no exact model cost")
            event_tick = int(item["tick"])
            self.history.append(
                EventV1(
                    tick=event_tick,
                    event_type="action_executed",
                    owner=owner,
                    card_id=card_id,
                    data={
                        "kind": "play_card",
                        "effective_card_id": card_id,
                        "effective_cost": cost,
                        "form_code": int(item["formCode"]),
                        "action_id": f"live/{self.generation}/{sequence}",
                    },
                )
            )

    def _settle_submissions(self, raw: Mapping[str, object], hand: tuple[int, ...]) -> None:
        tick = int(raw["tick"])
        self.receipts.clear()
        remaining = []
        for submitted in self.pending:
            slot = submitted.action.hand_slot
            card = submitted.action.card_id
            rotated = submitted.slot_changed or hand[slot] != card
            seen = submitted.event_seen or any(
                event["owner"] == 0 and event["cardId"] == card
                and event["sequence"] > submitted.source_cursor and event["tick"] >= submitted.requested_tick
                for event in raw["events"]
            )
            if rotated and seen:
                self.receipts.append((submitted, "confirmed"))
            elif raw["ended"]:
                self.receipts.append((submitted, "match_ended"))
            elif tick >= submitted.expected_tick + RECEIPT_TIMEOUT_TICKS:
                self.receipts.append((submitted, "timed_out"))
            else:
                remaining.append(replace(submitted, slot_changed=rotated, event_seen=seen))
        self.pending = remaining

    def _append_transitions(
        self, tick: int, towers: Sequence[TowerStateV1], entities: Sequence[EntityStateV1]
    ) -> None:
        current_entities = {entity.entity_id: entity for entity in entities}
        current_towers = {tower.entity_id: tower for tower in towers}
        for entity in entities:
            if entity.entity_id not in self.previous_entities:
                self.history.append(
                    EventV1(
                        tick=tick,
                        event_type="spawn",
                        owner=entity.owner,
                        entity_id=entity.entity_id,
                        card_id=entity.card_id,
                        position=entity.position,
                    )
                )
        for entity_id, previous in self.previous_entities.items():
            if entity_id not in current_entities:
                self.history.append(
                    EventV1(
                        tick=tick,
                        event_type="death_or_despawn",
                        owner=previous.owner,
                        entity_id=entity_id,
                        card_id=previous.card_id,
                        position=previous.position,
                    )
                )
        for item in (*entities, *towers):
            previous = (
                self.previous_towers.get(item.entity_id)
                if isinstance(item, TowerStateV1)
                else self.previous_entities.get(item.entity_id)
            )
            if previous is None or item.hitpoints is None or previous.hitpoints is None:
                continue
            if item.hitpoints < previous.hitpoints:
                amount = previous.hitpoints - item.hitpoints
                combat, provenance = _fair_snapshot_damage_contract(
                    tick=tick,
                    target_entity=item.entity_id,
                    target_card_id=(None if isinstance(item, TowerStateV1) else item.card_id),
                    position=item.position,
                    amount=amount,
                )
                self.history.append(
                    EventV1(
                        tick=tick,
                        event_type="tower_damage" if isinstance(item, TowerStateV1) else "damage",
                        owner=item.owner,
                        entity_id=item.entity_id,
                        card_id=None if isinstance(item, TowerStateV1) else item.card_id,
                        position=item.position,
                        combat=combat,
                        runtime_provenance=provenance,
                        data={"amount": amount, "remaining_hp": item.hitpoints},
                    )
                )
            if isinstance(item, TowerStateV1) and previous.active and not item.active:
                self.history.append(
                    EventV1(
                        tick=tick,
                        event_type="tower_destroyed",
                        owner=item.owner,
                        entity_id=item.entity_id,
                        position=item.position,
                        data={"tower_kind": item.tower_kind},
                    )
                )
        self.previous_entities = current_entities
        self.previous_towers = current_towers

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
            spec = self.bundle.card_specs.get(entity.card_id)
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

    def _action_mask(
        self,
        tick: int,
        own: Mapping[str, object],
        towers: Sequence[TowerStateV1],
        entities: Sequence[EntityStateV1],
    ) -> ActionMaskV1:
        elixir_raw = int(own["elixirRaw"])
        reserved_raw = int(round(sum(pending.cost for pending in self.pending if not pending.slot_changed) * 10_000))
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
        reserved_cards = {pending.action.card_id for pending in self.pending}
        for slot, card_id in enumerate(hand):
            spec = self.bundle.card_specs[card_id]
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
            rows = artifact.rows
            first_bin = _earliest_affordable_bin(int(round(float(spec.elixir_cost) * 10_000)), window)
            legal = card_id not in reserved_cards and first_bin is not None and any(any(row) for row in rows)
            slots.append(legal)
            reasons[str(slot)] = "legal" if legal else "pending" if card_id in reserved_cards else "insufficient_elixir"
            if first_bin is not None:
                affordable_from[str(slot)] = first_bin
            placements[str(slot)] = FrozenMapping(
                {
                    "card_id": card_id,
                    "visible_card_id": card_id,
                    "effective_card_id": card_id,
                    "native_effective_card_id": card_id,
                    "effective_cost": float(spec.elixir_cost),
                    "form_code": 0,
                    "native_form_code": 0,
                    "source_native_hand_slot": slot,
                    "shape": (32, 18),
                    "row_major": rows,
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

    def record_actions(self, actions: Sequence[ActionV1], requested_tick: int, source_cursor: int) -> None:
        for action in actions:
            if action.kind is not ActionKind.PLAY_CARD or action.card_id is None:
                continue
            if action.hand_slot is None or not 0 <= action.hand_slot < 4:
                raise LiveLadderError("submitted card has no valid hand slot")
            cost = self.bundle.card_specs[int(action.card_id)].elixir_cost
            if cost is None:
                raise LiveLadderError("submitted card has no exact cost")
            self.pending.append(
                PendingTouch(
                    action=action,
                    cost=float(cost),
                    requested_tick=requested_tick,
                    expected_tick=requested_tick + max(1, int(action.execute_offset_ticks or 1)) + COMMAND_EXECUTION_DELAY_TICKS,
                    source_cursor=source_cursor,
                )
            )

    def build(self, value: Mapping[str, object]) -> ObservationV1:
        raw = validate_fair_payload(value)
        generation = int(raw["generation"])
        tick = int(raw["tick"])
        if generation != self.generation:
            self._reset(generation)
        if tick < self.last_tick:
            raise LiveLadderError("live tick moved backwards within one battle generation")
        self._append_probe_events(raw)
        towers = self._towers(raw)
        entities = self._entities(raw, tick)
        self._append_transitions(tick, towers, entities)
        while self.history and self.history[0].tick < tick - 900:
            self.history.popleft()

        players = raw["players"]
        assert isinstance(players, list)
        own = players[0]
        enemy = players[1]
        assert isinstance(own, Mapping) and isinstance(enemy, Mapping)
        hand = tuple(int(card) for card in own["hand"])
        self._settle_submissions(raw, hand)
        deck = tuple(int(card) for card in own["deck"])
        cycle = tuple(int(card) for card in own["cycle"])
        local = PlayerStateV1(
            owner=0,
            crowns=int(own["crownsRaw"]),
            tower_ids=tuple(tower.entity_id for tower in towers if tower.owner == 0),
            elixir_exact=int(own["elixirRaw"]) / 10_000.0,
            elixir_visible=int(own["elixirRaw"]) / 10_000.0,
            hand=hand,
            next_card=(int(own["nextCard"]) if own.get("nextCard") is not None else None),
            deck=deck,
            cycle=cycle,
            private_state_visible=True,
            metadata={
                "hand_slot_by_card": {str(card): slot for slot, card in enumerate(hand)},
                "hand_runtime_by_slot": {str(slot): {"form_code": 0} for slot in range(len(hand))},
            },
        )
        opponent = PlayerStateV1(
            owner=1,
            crowns=int(enemy["crownsRaw"]),
            tower_ids=tuple(tower.entity_id for tower in towers if tower.owner == 1),
            revealed_cards=tuple(sorted(self.opponent_revealed)),
            private_state_visible=False,
        )
        action_mask = self._action_mask(tick, own, towers, entities)
        pending = tuple(
            PendingActionV1(
                action=item.action,
                requested_tick=item.requested_tick,
                expected_execution_tick=item.expected_tick,
            )
            for item in self.pending if not item.slot_changed
        )
        phase, multiplier = self._phase(tick)
        ended = bool(raw["ended"])
        winner = None
        results = (0.0, 0.0)
        if ended and local.crowns != opponent.crowns:
            winner = 0 if local.crowns > opponent.crowns else 1
            results = (1.0, -1.0) if winner == 0 else (-1.0, 1.0)
        terminal = TerminalV1(
            ended=ended,
            winner=winner,
            result_by_owner=results,
            reason="native_finalized" if ended else None,
            terminal_tick=tick if ended else None,
        )
        observation = ObservationV1(
            tier=ObservationTier.FAIR,
            owner=0,
            tick=tick,
            episode_id=self.episode_id,
            ruleset_id=RULESET_ID,
            phase=phase,
            time=TimeStateV1(
                elapsed_ms=tick * TICK_MS,
                remaining_ms=max(0, NATIVE_GAMEPLAY_END_TICK - tick) * TICK_MS,
                capture_time_ms=int(time.time() * 1000),
                available_time_ms=int(time.time() * 1000),
                elixir_multiplier=multiplier,
                tick_ms=TICK_MS,
            ),
            players=(local, opponent),
            towers=towers,
            entities=entities,
            events=tuple(self.history),
            action_mask=action_mask,
            pending_actions=pending,
            terminal=terminal,
            truncated=False,
            metadata={
                "source_schema": EXPECTED_SCHEMA,
                "generation": generation,
                "source_sequence": int(raw["sequence"]),
                "opponent_private_state_redacted": True,
                "entity_overflow_count": 0,
            },
        )
        self.last_tick = tick
        return observation
