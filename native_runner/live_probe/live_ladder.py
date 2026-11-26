from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from ..arena import OccupiedFootprintV1, card_placement_mask, native_building_footprint
from ..battle_env import (
    ELIXIR_SEGMENTS,
    TOWER_LAYOUT,
    _action_touch_delay_bin,
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
    TargetKind,
    TerminalV1,
    TimeStateV1,
    TowerStateV1,
)
from ..cr_native_env import COMMAND_EXECUTION_DELAY_TICKS
from ..match_factory import NATIVE_GAMEPLAY_END_TICK
from ..training.card_features import HERO_FORM_TO_BASE_CARD
from ..training.v4.catalog import UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
from ..training.v4.factory import build_live_tensorizer_v4, production_semantic_bundle
from ..training.v4.expert import POLICY_DECISION_TICKS
from ..training.v4.policy_session import PolicySessionV4, load_policy_v4
from .external_memory import ExternalMemoryError
from .external_source import (
    DEFAULT_READER_PORT,
    EXPECTED_READER_SHA256,
    AdbExternalReaderRuntime,
    ExternalLiveClient,
    ExternalSourceError,
)
from .live_actions import (
    AdbTouchController,
    LivePlayDispatcher,
    TouchLayout,
    _observe_with_snapshot_retry,
)
from .live_contracts import (
    _exact_int,
    LiveLadderError,
    authorization_open,
    require_authorization,
    sha256_file,
    validate_fair_payload,
    validate_status,
)
from .live_observation import LiveObservationBuilder, PendingTouch
from .live_settings import (
    AUTHORIZATION_END_EPOCH_EXCLUSIVE,
    AUTHORIZATION_END_EXCLUSIVE,
    AUTHORIZATION_START,
    DEFAULT_CHECKPOINT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    EPISODE_PREFIX,
    EXPECTED_LIBG_BUILD_ID,
    EXPECTED_LIBG_SHA256,
    EXPECTED_PACKAGE,
    EXPECTED_PROTOCOL,
    EXPECTED_SCHEMA,
    EXPECTED_STATUS_SCHEMA,
    EXPECTED_VERSION,
    HOG_26_SET,
    EXPECTED_CHECKPOINT_SHA256,
    HOG_26_DECK,
    PREVIOUS_CHECKPOINT_SHA256,
    PARIS,
    RECEIPT_TIMEOUT_TICKS,
    RULESET_ID,
    SUPPORTED_TOWER_TROOPS,
    TOWER_POSITION,
    TICK_MS,
)


@dataclass(frozen=True, slots=True)
class LiveServices:

    runtime_factory: Callable[[argparse.Namespace], AdbExternalReaderRuntime] | None = None
    client_factory: Callable[..., ExternalLiveClient] | None = None
    policy_loader: Callable[..., Any] | None = None
    tensorizer_factory: Callable[..., Any] | None = None
    session_factory: Callable[..., PolicySessionV4] | None = None
    observation_builder_factory: Callable[[], LiveObservationBuilder] | None = None
    touch_controller_factory: Callable[..., AdbTouchController] | None = None
    dispatcher_factory: Callable[..., LivePlayDispatcher] | None = None


def _checkpoint(path: Path) -> Path:
    actual = path.resolve()
    if not actual.is_file():
        raise LiveLadderError(f"checkpoint does not exist: {actual}")
    digest = sha256_file(actual)
    allowed = {EXPECTED_CHECKPOINT_SHA256}
    if actual.name.casefold() != DEFAULT_CHECKPOINT.name.casefold():
        allowed.add(PREVIOUS_CHECKPOINT_SHA256)
    if digest not in allowed:
        raise LiveLadderError(f"checkpoint SHA-256 {digest} does not match a pinned policy for {actual.name}")
    return actual

def _write_json_line(stream: Any, value: Mapping[str, object]) -> None:
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()

def _live_card_costs() -> dict[int, int]:
    bundle = production_semantic_bundle()
    result: dict[int, int] = {}
    for card_id, specification in bundle.card_specs.items():
        if specification.elixir_cost is None:
            continue
        cost = float(specification.elixir_cost)
        if not cost.is_integer():
            raise LiveLadderError(f"card {card_id} has a non-integral live elixir cost")
        result[int(card_id)] = int(cost)
    return result

def _external_runtime(args: argparse.Namespace) -> AdbExternalReaderRuntime:
    return AdbExternalReaderRuntime(
        adb=args.adb,
        serial=args.serial,
        reader_binary=args.reader,
        host=args.host,
        port=args.port,
    )

def _arm_until_battle(client: ExternalLiveClient, account_id: int, wait_seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while True:
        try:
            client.arm(account_id)
            return
        except ExternalMemoryError as error:
            if time.monotonic() >= deadline:
                raise LiveLadderError(f"could not identify an active attested Hog 2.6 battle: {error}") from error
            time.sleep(0.25)
        except ExternalSourceError:
            raise

def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise LiveLadderError("CUDA was requested but no CUDA device is available")
    return device

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

def run_policy(args: argparse.Namespace, *, services: LiveServices | None = None) -> dict[str, object]:
    dependencies = services or LiveServices()
    runtime_factory = dependencies.runtime_factory or _external_runtime
    client_factory = dependencies.client_factory or ExternalLiveClient
    policy_loader = dependencies.policy_loader or load_policy_v4
    tensorizer_factory = dependencies.tensorizer_factory or build_live_tensorizer_v4
    session_factory = dependencies.session_factory or PolicySessionV4
    builder_factory = dependencies.observation_builder_factory or LiveObservationBuilder
    controller_factory = dependencies.touch_controller_factory or AdbTouchController
    dispatcher_factory = dependencies.dispatcher_factory or LivePlayDispatcher
    require_authorization()
    if args.max_decisions is not None and args.max_decisions < 1:
        raise LiveLadderError("--max-decisions must be positive")
    if args.max_plays is not None and args.max_plays < 1:
        raise LiveLadderError("--max-plays must be positive")
    if args.threads < 1:
        raise LiveLadderError("--threads must be positive")
    if args.start_tick_grace < 0:
        raise LiveLadderError("--start-tick-grace must be non-negative")
    torch.set_num_threads(int(args.threads))
    device = _resolve_device(args.device)
    checkpoint = _checkpoint(args.checkpoint)
    loaded = policy_loader(checkpoint, device=device, delay_policy="earliest")
    tensorizer = tensorizer_factory(HOG_26_DECK, actor_owner=0, own_form_availability=(0,) * 8)
    session = session_factory(
        loaded.model,
        tensorizer,
        device=device,
        sample=False,
        validate_tensors=bool(args.validate_tensors),
    )
    runtime = runtime_factory(args)
    client: ExternalLiveClient | None = None
    decisions = 0
    plays = 0
    prepare_ms: float | None = None
    log_path = args.log.resolve()
    try:
        runtime.start()
        client = client_factory(runtime, local_account_id=args.account_id, card_costs=_live_card_costs())
        validate_status(client.status())
        print(
            json.dumps(
                {
                    "schema": "nullsroyale-live-ready.v1",
                    "device": str(device),
                    "waiting_for_battle": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _arm_until_battle(client, args.account_id, args.battle_wait_seconds)
        status = validate_status(client.status())
        if status.get("snapshotReady") is not True or status.get("hog26DeckMatch") is not True:
            raise LiveLadderError("the discovered local battle does not use the attested Hog 2.6 deck")
        controller = None
        if not args.shadow:
            if args.layout is None:
                raise LiveLadderError("touch-enabled play requires --layout")
            controller = controller_factory(args.adb, args.serial, TouchLayout.load(args.layout))
            controller.verify()
        builder = builder_factory()
        dispatcher = dispatcher_factory(builder, controller) if controller is not None else None
        cursor = 0
        last_sequence = 0
        last_tick_progress = time.monotonic()
        last_decision_tick = -10_000
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            while True:
                require_authorization()
                raw = _observe_with_snapshot_retry(client, cursor)
                if builder.generation is not None and int(raw["generation"]) != builder.generation:
                    raise LiveLadderError("battle generation changed during the policy session")
                if builder.generation is None or int(raw["tick"]) > builder.last_tick:
                    last_tick_progress = time.monotonic()
                elif builder.pending and time.monotonic() - last_tick_progress > RECEIPT_TIMEOUT_TICKS * TICK_MS / 1000:
                    raise LiveLadderError("battle clock stopped before the submitted play was confirmed")
                if int(raw["sequence"]) == last_sequence:
                    time.sleep(0.02)
                    continue
                last_sequence = int(raw["sequence"])
                cursor = int(raw["eventCursor"])
                observation = builder.build(raw)
                for submission, status in builder.receipts:
                    action = submission.action
                    _write_json_line(log, {
                        "schema": "nullsroyale-live-execution.v1", "generation": builder.generation,
                        "requested_tick": submission.requested_tick, "observed_tick": observation.tick,
                        "confirmed_tick": observation.tick if status == "confirmed" else None,
                        "card_id": action.card_id, "hand_slot": action.hand_slot, "target_grid": action.target_grid,
                        "confirmed": status == "confirmed", "status": status,
                    })
                    plays += int(status == "confirmed")
                if any(status == "timed_out" for _, status in builder.receipts):
                    raise LiveLadderError("a submitted play exceeded the receipt deadline")
                if not session._episode_started:
                    late_by = observation.tick - args.maximum_start_tick
                    if late_by > 0 and (
                        late_by > args.start_tick_grace or not _startup_snapshot_is_pristine(raw)
                    ):
                        raise LiveLadderError(
                            f"first observation arrived at tick {observation.tick}; maximum is "
                            f"{args.maximum_start_tick} with {args.start_tick_grace} pristine grace ticks"
                        )
                    initial_own = observation.players[0].elixir_exact
                    if initial_own is None:
                        raise LiveLadderError("first live observation has no exact own elixir")


                    session.start_episode(observation, initial_elixir={0: initial_own, 1: initial_own})
                    if device.type == "cuda" and not args.validate_tensors:
                        prepare_started = time.perf_counter_ns()
                        session.observe(observation)
                        torch.cuda.synchronize(device)
                        prepare_ms = (time.perf_counter_ns() - prepare_started) / 1_000_000.0
                        _write_json_line(
                            log,
                            {
                                "schema": "nullsroyale-live-prepare.v1",
                                "generation": int(raw["generation"]),
                                "tick": observation.tick,
                                "startup_grace_ticks": max(0, late_by),
                                "device": str(device),
                                "prepare_ms": prepare_ms,
                                "cuda_graph": True,
                            },
                        )


                        last_decision_tick = observation.tick
                        continue
                if dispatcher is not None and dispatcher.follow_up is not None:
                    follow_up = dispatcher.follow_up[0]
                    status = dispatcher.advance(observation, cursor)
                    if status in ("submitted", "scheduled"):
                        continue
                    tensorizer.retain_previous_action_prefix(1)
                    _write_json_line(log, {"schema": "nullsroyale-follow-up-cancelled.v1", "tick": observation.tick,
                                           "card_id": follow_up.card_id, "reason": status})
                if observation.terminal.ended:
                    break
                at_decision_limit = args.max_decisions is not None and decisions >= args.max_decisions
                at_play_limit = dispatcher is not None and args.max_plays is not None and dispatcher.submitted >= args.max_plays
                if at_decision_limit or at_play_limit:
                    if builder.pending:
                        continue
                    break
                if observation.tick < last_decision_tick + POLICY_DECISION_TICKS:
                    continue
                decision = session.decide(observation, include_scores=True)
                actions = tuple(decision.decoded.actions)
                _write_json_line(
                    log,
                    {
                        "schema": "nullsroyale-live-decision.v1",
                        "authorization_end_utc_exclusive": AUTHORIZATION_END_EPOCH_EXCLUSIVE,
                        "generation": int(raw["generation"]),
                        "tick": observation.tick,
                        "checkpoint_sha256": loaded.checkpoint_sha256,
                        "device": str(device),
                        "inference_ms": decision.inference_ms,
                        "decision_ms": decision.decision_ms,
                        "scores": decision.scores,
                        "actions": [action.to_dict() for action in actions],
                        "shadow": bool(args.shadow),
                    },
                )
                decisions += 1
                if dispatcher is not None:
                    retained = dispatcher.submit(observation, cursor, actions, args.max_plays)
                    if retained < sum(a.kind is ActionKind.PLAY_CARD for a in actions):
                        tensorizer.retain_previous_action_prefix(retained)
                last_decision_tick = observation.tick
                if args.max_decisions is not None and decisions >= args.max_decisions:
                    if not builder.pending:
                        break
                if args.shadow:
                    break
    finally:
        session.end_episode()
        if client is not None:
            client.disarm()
        runtime.close()
    return {
        "ok": True,
        "shadow": bool(args.shadow),
        "decisions": decisions,
        "plays": plays,
        "submitted_plays": dispatcher.submitted if dispatcher is not None else 0,
        "device": str(device),
        "prepare_ms": prepare_ms,
        "log": str(log_path),
        "checkpoint_sha256": loaded.checkpoint_sha256,
    }

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Null's Royale live adapter.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument(
        "--reader",
        type=Path,
        default=Path("native_runner/live_probe/out/nullsroyale-memory-reader"),
    )
    parser.add_argument("--account-id", type=int, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    observe = sub.add_parser("observe")
    observe.add_argument("--cursor", type=int, default=0)
    observe.add_argument("--battle-wait-seconds", type=float, default=120.0)
    run = sub.add_parser("run")
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--device", default="auto")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--validate-tensors", action="store_true")
    run.add_argument("--shadow", action="store_true")
    run.add_argument("--layout", type=Path)
    run.add_argument("--log", type=Path, default=Path("captures/nullsroyale-live-decisions.jsonl"))
    run.add_argument("--maximum-start-tick", type=int, default=100)
    run.add_argument("--start-tick-grace", type=int, default=20)
    run.add_argument("--battle-wait-seconds", type=float, default=120.0)
    run.add_argument("--max-decisions", type=int)
    run.add_argument("--max-plays", type=int)
    return parser

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_policy(args)
    else:
        runtime = _external_runtime(args)
        client: ExternalLiveClient | None = None
        try:
            runtime.start()
            client = ExternalLiveClient(runtime, local_account_id=args.account_id, card_costs=_live_card_costs())
            if args.command == "status":
                result = validate_status(client.status())
            else:
                _arm_until_battle(client, args.account_id, args.battle_wait_seconds)
                result = client.observe(args.cursor)
        finally:
            if client is not None:
                client.disarm()
            runtime.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
