from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from ..training.v4.factory import build_live_tensorizer_v4, production_semantic_bundle
from ..training.v4.policy_session import LoadedPolicyV4, PolicySessionV4, load_policy_v4
from ..training.v4.tensorizer import UniversalObservationTensorizerV4
from .external_memory import ExternalMemoryError
from .external_source import AdbExternalReaderRuntime, ExternalLiveClient, _observe_with_snapshot_retry
from .live_actions import AdbTouchController, LivePlayDispatcher, TouchLayout, validate_touch_actions
from .live_contracts import (
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
    EXPECTED_CHECKPOINT_SHA256,
    HOG_26_DECK,
    PREVIOUS_CHECKPOINT_SHA256,
    RECEIPT_TIMEOUT_TICKS,
    TICK_MS,
)
from .policy_match import MatchFrame, MatchOptions, PolicyMatch, _startup_snapshot_is_pristine


@dataclass(frozen=True, slots=True)
class LiveServices:
    runtime_factory: Callable[[argparse.Namespace], AdbExternalReaderRuntime] | None = None
    client_factory: Callable[..., ExternalLiveClient] | None = None
    policy_loader: Callable[..., LoadedPolicyV4] | None = None
    tensorizer_factory: Callable[..., UniversalObservationTensorizerV4] | None = None
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
        quick = time.monotonic() < deadline
        try:
            client.arm(account_id, quick=quick)
            return
        except ExternalMemoryError as error:
            if not quick:
                raise LiveLadderError(f"could not identify an active attested Hog 2.6 battle: {error}") from error
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise LiveLadderError("CUDA was requested but no CUDA device is available")
    return device


@contextmanager
def _connected_client(args: argparse.Namespace, services: LiveServices) -> Iterator[ExternalLiveClient]:
    runtime = (services.runtime_factory or _external_runtime)(args)
    with ExitStack() as cleanup:
        cleanup.callback(runtime.close)
        runtime.start()
        factory = services.client_factory or ExternalLiveClient
        client = factory(runtime, local_account_id=args.account_id, card_costs=_live_card_costs())
        cleanup.callback(client.disarm)
        yield client


def _wait_for_match(client: ExternalLiveClient, args: argparse.Namespace, device: torch.device) -> None:
    validate_status(client.status())
    ready = {"schema": "nullsroyale-live-ready.v1", "device": str(device), "waiting_for_battle": True}
    print(json.dumps(ready, sort_keys=True), flush=True)
    _arm_until_battle(client, args.account_id, args.battle_wait_seconds)
    status = validate_status(client.status())
    if status.get("snapshotReady") is not True or status.get("hog26DeckMatch") is not True:
        raise LiveLadderError("the discovered local battle does not use the attested Hog 2.6 deck")


def _dispatcher(
    args: argparse.Namespace,
    services: LiveServices,
    builder: LiveObservationBuilder,
) -> LivePlayDispatcher | None:
    if args.shadow:
        return None
    if args.layout is None:
        raise LiveLadderError("touch-enabled play requires --layout")
    controller_factory = services.touch_controller_factory or AdbTouchController
    controller = controller_factory(args.adb, args.serial, TouchLayout.load(args.layout))
    controller.verify()
    return (services.dispatcher_factory or LivePlayDispatcher)(builder, controller)


def _frames(client: ExternalLiveClient, builder: LiveObservationBuilder) -> Iterator[MatchFrame]:
    cursor = 0
    last_sequence = 0
    last_progress = time.monotonic()
    timeout = RECEIPT_TIMEOUT_TICKS * TICK_MS / 1000
    while True:
        require_authorization()
        raw = _observe_with_snapshot_retry(client, cursor)
        if builder.generation is not None and int(raw["generation"]) != builder.generation:
            raise LiveLadderError("battle generation changed during the policy session")
        if builder.generation is None or int(raw["tick"]) > builder.last_tick:
            last_progress = time.monotonic()
        elif builder.pending and time.monotonic() - last_progress > timeout:
            raise LiveLadderError("battle clock stopped before the submitted play was confirmed")
        sequence = int(raw["sequence"])
        if sequence == last_sequence:
            time.sleep(0.02)
            continue
        last_sequence = sequence
        cursor = int(raw["eventCursor"])
        yield MatchFrame(raw, builder.build(raw))


def _match_options(args: argparse.Namespace) -> MatchOptions:
    if args.max_decisions is not None and args.max_decisions < 1:
        raise LiveLadderError("--max-decisions must be positive")
    if args.max_plays is not None and args.max_plays < 1:
        raise LiveLadderError("--max-plays must be positive")
    if args.threads < 1:
        raise LiveLadderError("--threads must be positive")
    if args.start_tick_grace < 0:
        raise LiveLadderError("--start-tick-grace must be non-negative")
    return MatchOptions(
        maximum_start_tick=args.maximum_start_tick,
        start_tick_grace=args.start_tick_grace,
        max_decisions=args.max_decisions,
        max_plays=args.max_plays,
        shadow=bool(args.shadow),
        validate_tensors=bool(args.validate_tensors),
    )


def run_policy(args: argparse.Namespace, *, services: LiveServices | None = None) -> dict[str, object]:
    require_authorization()
    options = _match_options(args)
    services = services or LiveServices()
    torch.set_num_threads(int(args.threads))
    device = _resolve_device(args.device)
    load_policy = services.policy_loader or load_policy_v4
    loaded = load_policy(_checkpoint(args.checkpoint), device=device, delay_policy="earliest")
    tensorizer_factory = services.tensorizer_factory or build_live_tensorizer_v4
    tensorizer = tensorizer_factory(HOG_26_DECK, actor_owner=0, own_form_availability=(0,) * 8)
    session = (services.session_factory or PolicySessionV4)(
        loaded.model, tensorizer, device=device, sample=False, validate_tensors=options.validate_tensors
    )
    with ExitStack() as cleanup:
        cleanup.callback(session.end_episode)
        client = cleanup.enter_context(_connected_client(args, services))
        _wait_for_match(client, args, device)
        builder = (services.observation_builder_factory or LiveObservationBuilder)()
        dispatcher = _dispatcher(args, services, builder)
        log_path = args.log.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            match = PolicyMatch(options, session, tensorizer, device, loaded.checkpoint_sha256, builder, dispatcher, log)
            return {**match.run(_frames(client, builder)), "log": str(log_path)}


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
        with _connected_client(args, LiveServices()) as client:
            if args.command == "status":
                result = validate_status(client.status())
            else:
                _arm_until_battle(client, args.account_id, args.battle_wait_seconds)
                result = client.observe(args.cursor)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
