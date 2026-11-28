from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence

from ..contracts import (
    ActionKind,
    ActionV1,
    EventV1,
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
from ..training.v4.factory import production_semantic_bundle
from .battlefield import Battlefield
from .live_contracts import LiveLadderError, validate_fair_payload
from .live_settings import EPISODE_PREFIX, EXPECTED_SCHEMA, RECEIPT_TIMEOUT_TICKS, RULESET_ID, TICK_MS
from .placement import PlacementMaskBuilder
from .play_ledger import PendingTouch, PlayLedger


class LiveObservationBuilder:
    def __init__(self) -> None:
        self.bundle = production_semantic_bundle()
        self.placement = PlacementMaskBuilder(self.bundle.card_specs)
        self.generation: int | None = None
        self.episode_id: str | None = None
        self.last_tick = 0
        self.battlefield = Battlefield(self.bundle)
        self.ledger = PlayLedger(COMMAND_EXECUTION_DELAY_TICKS, RECEIPT_TIMEOUT_TICKS)
        self.history: deque[EventV1] = deque()
        self.seen_public_events: set[int] = set()
        self.opponent_revealed: set[int] = set()

    @property
    def pending(self) -> list[PendingTouch]:
        return self.ledger.pending

    @property
    def receipts(self) -> list[tuple[PendingTouch, str]]:
        return self.ledger.receipts

    def _reset(self, generation: int) -> None:
        self.generation = generation
        self.episode_id = f"{EPISODE_PREFIX}-{generation}"
        self.last_tick = 0
        self.battlefield = Battlefield(self.bundle)
        self.ledger.pending.clear()
        self.ledger.receipts.clear()
        self.history.clear()
        self.seen_public_events.clear()
        self.opponent_revealed.clear()

    @staticmethod
    def _phase(tick: int) -> tuple[str, float]:
        multiplier = 1.0 if tick < 2400 else 2.0 if tick < 4800 else 3.0
        phase = "regulation" if tick < 3600 else "overtime" if tick < NATIVE_GAMEPLAY_END_TICK else "tiebreak"
        return phase, multiplier

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

    def record_actions(self, actions: Sequence[ActionV1], requested_tick: int, source_cursor: int) -> None:
        for action in actions:
            if action.kind is not ActionKind.PLAY_CARD or action.card_id is None:
                continue
            if action.hand_slot is None or not 0 <= action.hand_slot < 4:
                raise LiveLadderError("submitted card has no valid hand slot")
            cost = self.bundle.card_specs[int(action.card_id)].elixir_cost
            if cost is None:
                raise LiveLadderError("submitted card has no exact cost")
            self.ledger.record(action, float(cost), requested_tick, source_cursor)

    def _outstanding_actions(self) -> tuple[PendingActionV1, ...]:
        return tuple(
            PendingActionV1(
                action=item.action,
                requested_tick=item.requested_tick,
                expected_execution_tick=item.expected_tick,
            )
            for item in self.pending
            if not item.slot_changed
        )

    def _players(
        self,
        own: Mapping[str, object],
        enemy: Mapping[str, object],
        hand: tuple[int, ...],
        towers: Sequence[TowerStateV1],
    ) -> tuple[PlayerStateV1, PlayerStateV1]:
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
        return local, opponent

    @staticmethod
    def _terminal(ended: bool, tick: int, local: PlayerStateV1, opponent: PlayerStateV1) -> TerminalV1:
        winner = None
        results = (0.0, 0.0)
        if ended and local.crowns != opponent.crowns:
            winner = 0 if local.crowns > opponent.crowns else 1
            results = (1.0, -1.0) if winner == 0 else (-1.0, 1.0)
        return TerminalV1(
            ended=ended,
            winner=winner,
            result_by_owner=results,
            reason="native_finalized" if ended else None,
            terminal_tick=tick if ended else None,
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
        frame = self.battlefield.update(raw, tick, self.last_tick)
        self.history.extend(frame.transitions)
        while self.history and self.history[0].tick < tick - 900:
            self.history.popleft()

        players = raw["players"]
        assert isinstance(players, list)
        own = players[0]
        enemy = players[1]
        assert isinstance(own, Mapping) and isinstance(enemy, Mapping)
        hand = tuple(int(card) for card in own["hand"])
        self.ledger.settle(tick, raw["ended"], hand, raw["events"])
        local, opponent = self._players(own, enemy, hand, frame.towers)
        action_mask = self.placement.build(tick, own, self.pending, frame.towers, frame.entities)
        phase, multiplier = self._phase(tick)
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
            towers=frame.towers,
            entities=frame.entities,
            events=tuple(self.history),
            action_mask=action_mask,
            pending_actions=self._outstanding_actions(),
            terminal=self._terminal(bool(raw["ended"]), tick, local, opponent),
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
