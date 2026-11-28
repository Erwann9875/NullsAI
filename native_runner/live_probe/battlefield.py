from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..battle_env import TOWER_LAYOUT, _fair_snapshot_damage_contract
from ..contracts import EntityStateV1, EventV1, TowerStateV1
from ..training.card_features import HERO_FORM_TO_BASE_CARD
from ..training.v4.catalog import UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
from ..training.v4.factory import ProductionSemanticBundleV4
from .live_contracts import LiveLadderError
from .live_settings import TICK_MS, TOWER_POSITION


@dataclass(frozen=True, slots=True)
class BattlefieldFrame:
    towers: tuple[TowerStateV1, ...]
    entities: tuple[EntityStateV1, ...]
    transitions: Sequence[EventV1]


class Battlefield:
    def __init__(self, bundle: ProductionSemanticBundleV4) -> None:
        self.bundle = bundle
        self.first_seen: dict[int, int] = {}
        self.previous_entities: dict[int, EntityStateV1] = {}
        self.previous_towers: dict[int, TowerStateV1] = {}
        self.tower_registry: dict[tuple[int, str], tuple[int, tuple[int, int], float]] = {}

    def update(self, raw: Mapping[str, object], tick: int, previous_tick: int) -> BattlefieldFrame:
        towers = self._towers(raw)
        entities = self._entities(raw, tick, tick - previous_tick)
        return BattlefieldFrame(towers, entities, self._transitions(tick, towers, entities))

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
            _, kind = key
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
        if len(self.tower_registry) != len(TOWER_LAYOUT):
            raise LiveLadderError("the first live frame does not contain all six towers")
        towers: list[TowerStateV1] = []
        for _, owner, kind, _, _ in TOWER_LAYOUT:
            key = (owner, kind)
            item = current.get(key)
            registry = self.tower_registry.get(key)
            if registry is None:
                raise LiveLadderError(f"tower identity {key} was never observed")
            entity_id, position, maximum = registry
            hp = 0.0 if item is None else float(item["hp"])
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
                    visible_target=None,
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

    def _entities(self, raw: Mapping[str, object], tick: int, dt: int) -> tuple[EntityStateV1, ...]:
        result: list[EntityStateV1] = []
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

    @staticmethod
    def _entity_event(tick: int, kind: str, entity: EntityStateV1) -> EventV1:
        return EventV1(
            tick=tick,
            event_type=kind,
            owner=entity.owner,
            entity_id=entity.entity_id,
            card_id=entity.card_id,
            position=entity.position,
        )

    def _transitions(
        self, tick: int, towers: Sequence[TowerStateV1], entities: Sequence[EntityStateV1]
    ) -> list[EventV1]:
        events: list[EventV1] = []
        current_entities = {entity.entity_id: entity for entity in entities}
        current_towers = {tower.entity_id: tower for tower in towers}
        events.extend(
            self._entity_event(tick, "spawn", entity)
            for entity in entities
            if entity.entity_id not in self.previous_entities
        )
        events.extend(
            self._entity_event(tick, "death_or_despawn", previous)
            for entity_id, previous in self.previous_entities.items()
            if entity_id not in current_entities
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
                events.append(
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
                events.append(
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
        return events
