from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TextIO

import torch

from ..contracts import ActionKind, ObservationV1
from ..training.v4.expert import POLICY_DECISION_TICKS
from ..training.v4.policy_session import PolicySessionV4
from ..training.v4.tensorizer import UniversalObservationTensorizerV4
from .live_actions import LivePlayDispatcher
from .live_contracts import LiveLadderError
from .live_observation import LiveObservationBuilder
from .game_profile import AUTHORIZATION_END_EPOCH_EXCLUSIVE


def _startup_snapshot_is_pristine(raw: Mapping[str, object]) -> bool:
    events = raw.get("events")
    objects = raw.get("objects")
    if not isinstance(events, list) or not isinstance(objects, list) or events:
        return False
    return not any(
        isinstance(item, Mapping)
        and isinstance(item.get("cardId"), int)
        and not isinstance(item.get("cardId"), bool)
        and int(item["cardId"]) > 0
        for item in objects
    )


@dataclass(frozen=True, slots=True)
class MatchOptions:
    maximum_start_tick: int
    start_tick_grace: int
    max_decisions: int | None
    max_plays: int | None
    shadow: bool
    validate_tensors: bool


@dataclass(frozen=True, slots=True)
class MatchFrame:
    raw: Mapping[str, object]
    observation: ObservationV1

    @property
    def cursor(self) -> int:
        return int(self.raw["eventCursor"])


class PolicyMatch:
    def __init__(
        self,
        options: MatchOptions,
        session: PolicySessionV4,
        tensorizer: UniversalObservationTensorizerV4,
        device: torch.device,
        checkpoint_sha256: str,
        builder: LiveObservationBuilder,
        dispatcher: LivePlayDispatcher | None,
        log: TextIO,
    ) -> None:
        self.options = options
        self.session = session
        self.tensorizer = tensorizer
        self.device = device
        self.checkpoint_sha256 = checkpoint_sha256
        self.builder = builder
        self.dispatcher = dispatcher
        self.log = log
        self.decisions = 0
        self.confirmed = 0
        self.prepare_ms: float | None = None
        self.last_decision_tick: int | None = None
        self.started = session._episode_started

    def _log(self, schema: str, **fields: object) -> None:
        record = json.dumps({"schema": schema, **fields}, sort_keys=True, separators=(",", ":"))
        self.log.write(record + "\n")
        self.log.flush()

    def _record_receipts(self, observation: ObservationV1) -> None:
        for submission, outcome in self.builder.receipts:
            action = submission.action
            confirmed = outcome == "confirmed"
            self._log(
                "nullsroyale-live-execution.v1",
                generation=self.builder.generation,
                requested_tick=submission.requested_tick,
                observed_tick=observation.tick,
                confirmed_tick=observation.tick if confirmed else None,
                card_id=action.card_id,
                hand_slot=action.hand_slot,
                target_grid=action.target_grid,
                confirmed=confirmed,
                status=outcome,
            )
            self.confirmed += int(confirmed)
        if any(outcome == "timed_out" for _, outcome in self.builder.receipts):
            raise LiveLadderError("a submitted play exceeded the receipt deadline")

    def _start_episode(self, frame: MatchFrame) -> None:
        observation = frame.observation
        late_by = observation.tick - self.options.maximum_start_tick
        if late_by > 0 and (late_by > self.options.start_tick_grace or not _startup_snapshot_is_pristine(frame.raw)):
            raise LiveLadderError(
                f"first observation arrived at tick {observation.tick}; maximum is "
                f"{self.options.maximum_start_tick} with {self.options.start_tick_grace} pristine grace ticks"
            )
        elixir = observation.players[0].elixir_exact
        if elixir is None:
            raise LiveLadderError("first live observation has no exact own elixir")
        self.session.start_episode(observation, initial_elixir={0: elixir, 1: elixir})
        self.started = True
        if self.device.type == "cuda" and not self.options.validate_tensors:
            started = time.perf_counter_ns()
            self.session.observe(observation)
            torch.cuda.synchronize(self.device)
            self.prepare_ms = (time.perf_counter_ns() - started) / 1_000_000
            self.last_decision_tick = observation.tick
            self._log(
                "nullsroyale-live-prepare.v1",
                generation=int(frame.raw["generation"]),
                tick=observation.tick,
                startup_grace_ticks=max(0, late_by),
                device=str(self.device),
                prepare_ms=self.prepare_ms,
                cuda_graph=True,
            )

    def _advance_follow_up(self, frame: MatchFrame) -> bool:
        if self.dispatcher is None or self.dispatcher.follow_up is None:
            return False
        action = self.dispatcher.follow_up[0]
        outcome = self.dispatcher.advance(frame.observation, frame.cursor)
        if outcome in ("submitted", "scheduled"):
            return True
        self.tensorizer.retain_previous_action_prefix(1)
        self._log(
            "nullsroyale-follow-up-cancelled.v1",
            tick=frame.observation.tick,
            card_id=action.card_id,
            reason=outcome,
        )
        return False

    def _decision_limit_reached(self) -> bool:
        limit = self.options.max_decisions
        return limit is not None and self.decisions >= limit

    def _play_limit_reached(self) -> bool:
        limit = self.options.max_plays
        return self.dispatcher is not None and limit is not None and self.dispatcher.submitted >= limit

    def _decide(self, frame: MatchFrame) -> None:
        observation = frame.observation
        decision = self.session.decide(observation, include_scores=True)
        actions = tuple(decision.decoded.actions)
        self._log(
            "nullsroyale-live-decision.v1",
            authorization_end_utc_exclusive=AUTHORIZATION_END_EPOCH_EXCLUSIVE,
            generation=int(frame.raw["generation"]),
            tick=observation.tick,
            checkpoint_sha256=self.checkpoint_sha256,
            device=str(self.device),
            inference_ms=decision.inference_ms,
            decision_ms=decision.decision_ms,
            scores=decision.scores,
            actions=[action.to_dict() for action in actions],
            shadow=self.options.shadow,
        )
        self.decisions += 1
        if self.dispatcher is not None:
            retained = self.dispatcher.submit(observation, frame.cursor, actions, self.options.max_plays)
            if retained < sum(action.kind is ActionKind.PLAY_CARD for action in actions):
                self.tensorizer.retain_previous_action_prefix(retained)
        self.last_decision_tick = observation.tick

    def run(self, frames: Iterable[MatchFrame]) -> dict[str, object]:
        for frame in frames:
            observation = frame.observation
            self._record_receipts(observation)
            if not self.started:
                self._start_episode(frame)
                if self.prepare_ms is not None:
                    continue
            if self._advance_follow_up(frame):
                continue
            if observation.terminal.ended:
                break
            if self._decision_limit_reached() or self._play_limit_reached():
                if self.builder.pending:
                    continue
                break
            if self.last_decision_tick is not None and observation.tick < self.last_decision_tick + POLICY_DECISION_TICKS:
                continue
            self._decide(frame)
            if self.options.shadow or (self._decision_limit_reached() and not self.builder.pending):
                break
        return {
            "ok": True,
            "shadow": self.options.shadow,
            "decisions": self.decisions,
            "plays": self.confirmed,
            "submitted_plays": self.dispatcher.submitted if self.dispatcher else 0,
            "device": str(self.device),
            "prepare_ms": self.prepare_ms,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
