from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from native_runner.live_probe.live_ladder import (
    HOG_26_DECK,
    LiveLadderError,
    LiveObservationBuilder,
    LivePlayDispatcher,
    AdbTouchController,
    TouchLayout,
    _observe_with_snapshot_retry,
    _resolve_device,
    _startup_snapshot_is_pristine,
    authorization_open,
    validate_fair_payload,
    validate_touch_actions,
)
from native_runner.tests.live_fixtures import battle_payload
from native_runner.live_probe import live_contracts, live_ladder
from native_runner.contracts import ActionKind, ActionV1, TargetKind
from native_runner.training.v4.factory import build_live_tensorizer_v4
from native_runner.live_probe.external_memory import ExternalMemoryError


@pytest.fixture(autouse=True)
def authorized_clock(monkeypatch):
    clock = Mock(wraps=datetime)
    clock.now.side_effect = lambda tz: datetime(2026, 11, 28, 12, tzinfo=tz)
    monkeypatch.setattr(live_contracts, "datetime", clock)


def test_live_device_auto_prefers_cuda_and_falls_back_to_cpu() -> None:
    with patch("native_runner.live_probe.live_ladder.torch.cuda.is_available", return_value=True):
        assert str(_resolve_device("auto")) == "cuda"
    with patch("native_runner.live_probe.live_ladder.torch.cuda.is_available", return_value=False):
        assert str(_resolve_device("auto")) == "cpu"


def test_live_snapshot_retry_recovers_from_a_transient_consistency_race() -> None:
    class Client:
        calls = 0

        def observe(self, cursor: int) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                raise ExternalMemoryError("native state changed across three bounded snapshot attempts")
            return {"eventCursor": cursor}

    client = Client()
    with patch("native_runner.live_probe.live_ladder.time.sleep"):
        assert _observe_with_snapshot_retry(client, 7) == {"eventCursor": 7}
    assert client.calls == 2


def test_startup_grace_requires_no_public_card_play_or_deployed_unit() -> None:
    pristine = battle_payload()
    assert _startup_snapshot_is_pristine(pristine)

    with_event = deepcopy(pristine)
    with_event["events"] = [{"kind": "card_play", "cardId": HOG_26_DECK[0]}]
    assert not _startup_snapshot_is_pristine(with_event)

    with_unit = deepcopy(pristine)
    assert isinstance(with_unit["objects"], list)
    with_unit["objects"].append({"cardId": HOG_26_DECK[0]})
    assert not _startup_snapshot_is_pristine(with_unit)




def test_authorization_window_is_inclusive_through_december_first_paris() -> None:
    assert not authorization_open(datetime.fromisoformat("2026-11-22T23:59:59+01:00"))
    assert authorization_open(datetime.fromisoformat("2026-11-23T00:00:00+01:00"))
    assert authorization_open(datetime.fromisoformat("2026-12-01T23:59:59+01:00"))
    assert not authorization_open(datetime.fromisoformat("2026-12-02T00:00:00+01:00"))


def test_fair_boundary_rejects_opponent_private_state() -> None:
    payload = battle_payload()
    players = payload["players"]
    assert isinstance(players, list)
    players[1]["elixirRaw"] = 60_000
    with pytest.raises(LiveLadderError, match="opponent-private"):
        validate_fair_payload(payload)


def test_live_builder_tensorizes_unknown_opponent_without_private_fields() -> None:
    observation = LiveObservationBuilder().build(battle_payload())
    enemy = observation.players[1]
    assert enemy.private_state_visible is False
    assert enemy.elixir_exact is None
    assert enemy.hand == enemy.deck == enemy.cycle == ()
    tensorizer = build_live_tensorizer_v4(HOG_26_DECK, own_form_availability=(0,) * 8)
    tensorizer.start_episode(observation, initial_elixir={0: 5.0, 1: 5.0})
    try:
        batch = tensorizer.tensorize(observation, validate=True)
    finally:
        tensorizer.end_episode()
    assert int(batch.towers.mask.sum()) == 6
    assert int(batch.candidates.mask.sum()) == 4


def test_touch_layout_maps_owner_zero_world_to_bottom_screen() -> None:
    layout = TouchLayout(
        screen_width=1080,
        screen_height=1920,
        arena_left=0,
        arena_top=200,
        arena_right=1079,
        arena_bottom=1500,
        hand_y=1700,
        hand_centers_x=(135, 405, 675, 945),
        owner_zero_at_screen_bottom=True,
    )
    near_owner = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=0,
        card_id=HOG_26_DECK[0],
        target_kind=TargetKind.GRID,
        target_grid=(0, 0),
    )
    far_owner = deepcopy(near_owner.to_dict())
    far_owner["target_grid"] = (17, 31)
    far = ActionV1.from_mapping(far_owner)
    near_pixel = layout.target_pixel(near_owner)
    far_pixel = layout.target_pixel(far)
    assert near_pixel[0] < far_pixel[0]
    assert near_pixel[1] > far_pixel[1]


def test_touch_controller_drags_from_hand_to_exact_subcell_target() -> None:
    layout = TouchLayout(
        screen_width=900,
        screen_height=1600,
        arena_left=75,
        arena_top=160,
        arena_right=825,
        arena_bottom=1290,
        hand_y=1425,
        hand_centers_x=(282, 451, 619, 787),
    )
    controller = AdbTouchController(Path("adb"), "test", layout, rng=random.Random(7))
    action = ActionV1.play(
        owner=0,
        hand_slot=1,
        grid=(7, 14),
        card_id=HOG_26_DECK[4],
        subcell_offset=(0.25, -0.25),
    )
    target = layout.target_pixel(action)

    with patch.object(controller, "_run") as run:
        controller.execute((action,))

    command = run.call_args.args
    assert command[:4] == ("shell", "input", "touchscreen", "swipe")
    start = (int(command[4]), int(command[5]))
    release = (int(command[6]), int(command[7]))
    duration = int(command[8])
    assert abs(start[0] - 451) <= layout.hand_jitter_px
    assert abs(start[1] - 1425) <= layout.hand_jitter_px
    assert release == target
    assert layout.drag_min_ms <= duration <= layout.drag_max_ms
    assert target != layout.target_pixel(
        ActionV1.play(owner=0, hand_slot=1, grid=(7, 14), card_id=HOG_26_DECK[4])
    )


def test_touch_controller_bounds_release_jitter_inside_selected_tile() -> None:
    layout = TouchLayout(
        screen_width=900,
        screen_height=1600,
        arena_left=75,
        arena_top=160,
        arena_right=825,
        arena_bottom=1200,
        hand_y=1425,
        hand_centers_x=(282, 451, 619, 787),
    )
    controller = AdbTouchController(Path("adb"), "test", layout, rng=random.Random(7))
    action = ActionV1.play(
        owner=0,
        hand_slot=1,
        grid=(7, 14),
        card_id=HOG_26_DECK[4],
    )
    tile_center = layout.target_pixel(action)

    with patch.object(controller, "_run") as run:
        controller.execute((action,))

    command = run.call_args.args
    release = (int(command[6]), int(command[7]))
    assert abs(release[0] - tile_center[0]) <= layout.release_jitter_px
    assert abs(release[1] - tile_center[1]) <= layout.release_jitter_px
    assert release != tile_center


def test_touch_guard_rejects_a_card_that_does_not_match_the_live_slot() -> None:
    observation = LiveObservationBuilder().build(battle_payload())
    action = ActionV1.play(
        owner=0,
        hand_slot=0,
        grid=(8, 8),
        card_id=HOG_26_DECK[1],
    )

    with pytest.raises(LiveLadderError, match="does not match live hand slot"):
        validate_touch_actions(observation, (action,))


def _battle_frame(tick, *, hand=None, elixir=5, events=(), cursor=0, ended=False):
    raw = battle_payload()
    raw.update(tick=tick, sequence=tick + 1, events=list(events), eventCursor=cursor, ended=ended)
    own = raw["players"][0]
    own.update(elixirRaw=int(elixir * 10000), elixir=float(elixir))
    if hand is not None:
        own["hand"] = list(hand)
        own["cycle"] = [card for card in HOG_26_DECK if card not in hand]
        own["nextCard"] = own["cycle"][0]
    return raw


def _event(sequence, tick, slot, cost):
    return dict(sequence=sequence, kind="card_play", tick=tick, owner=0, cardId=HOG_26_DECK[slot],
                cost=cost, formCode=0)


def _play(slot, grid=(9, 10), offset=1):
    return ActionV1.play(owner=0, hand_slot=slot, grid=grid, card_id=HOG_26_DECK[slot], execute_offset_ticks=offset)


@pytest.mark.parametrize("compound", [False, True], ids=["next-decision", "same-decision"])
def test_policy_loop_submits_legal_followup_before_first_receipt(monkeypatch, tmp_path, compound):
    first, second = _play(1), _play(0, (4, 12), 4 if compound else 1)
    rotated = (HOG_26_DECK[4], *HOG_26_DECK[1:4])
    final_hand = (rotated[0], HOG_26_DECK[5], *rotated[2:])
    frames = iter((
        _battle_frame(100), _battle_frame(102), _battle_frame(105),
        _battle_frame(110, hand=rotated, elixir=4, events=[_event(1, 109, 0, 1)], cursor=1),
        _battle_frame(140, hand=final_hand, elixir=1, events=[_event(2, 138, 1, 4)], cursor=2),
    ))
    current_tick = 0

    def snapshot(_cursor):
        nonlocal current_tick
        frame = next(frames)
        current_tick = frame["tick"]
        return frame

    def decide(obs, **_kwargs):
        if obs.tick == 100:
            actions = (first, second) if compound else (first,)
        else:
            assert obs.tick == 105 and obs.action_mask.reasons["effective_elixir"] == 1
            assert not obs.action_mask.hand_slots[1]
            actions = (second,)
        return SimpleNamespace(decoded=SimpleNamespace(actions=actions), decision_ms=1, inference_ms=1, scores={})

    session = Mock()
    session._episode_started = True
    session.decide.side_effect = decide
    client = Mock()
    client.status.return_value = {"snapshotReady": True, "hog26DeckMatch": True}
    client.observe.side_effect = snapshot
    sent = []
    controller = Mock()
    controller.execute.side_effect = lambda actions: sent.append((current_tick, actions[0].card_id, actions[0].target_grid))
    runtime = Mock()
    services = live_ladder.LiveServices(
        runtime_factory=Mock(return_value=runtime),
        client_factory=Mock(return_value=client),
        policy_loader=Mock(return_value=SimpleNamespace(model=None, checkpoint_sha256="test")),
        session_factory=Mock(return_value=session),
        touch_controller_factory=Mock(return_value=controller),
    )
    monkeypatch.setattr(live_ladder, "validate_status", lambda value: value)
    monkeypatch.setattr(live_ladder, "_checkpoint", lambda path: path)
    log = tmp_path / "decisions.jsonl"
    layout = Path(__file__).resolve().parents[1] / "live_probe" / "mumu_900x1600.json"
    args = live_ladder._parser().parse_args([
        "--adb", "unused", "--serial", "unused", "--account-id", "1", "run",
        "--device", "cpu", "--layout", str(layout), "--log", str(log), "--max-plays", "2",
    ])
    summary = live_ladder.run_policy(args, services=services)

    client.disarm.assert_called_once_with()
    runtime.close.assert_called_once_with()
    session.end_episode.assert_called_once_with()
    assert sent == [(100, first.card_id, (9, 10)), (105, second.card_id, (4, 12))]
    records = (json.loads(line) for line in log.read_text().splitlines())
    receipts = [record for record in records if record.get("confirmed")]
    assert [(r["requested_tick"], r["confirmed_tick"]) for r in receipts] == [(105, 110), (100, 140)]
    assert summary["plays"] == summary["submitted_plays"] == 2


@pytest.mark.parametrize("actions,reason", [
    ((_play(1), _play(2)), "insufficient elixir"),
    ((_play(0), _play(0)), "distinct hand slots"),
])
def test_invalid_compound_spends_nothing(actions, reason):
    builder, controller = LiveObservationBuilder(), Mock()
    dispatcher = LivePlayDispatcher(builder, controller)
    with pytest.raises(LiveLadderError, match=reason):
        dispatcher.submit(builder.build(_battle_frame(100)), 0, actions, None)
    controller.execute.assert_not_called()
    assert not builder.pending


def test_changed_hand_cancels_followup_and_terminal_settles_pending():
    builder, controller = LiveObservationBuilder(), Mock()
    dispatcher = LivePlayDispatcher(builder, controller)
    dispatcher.submit(builder.build(_battle_frame(100)), 0, (_play(1), _play(0, offset=3)), None)
    changed = (HOG_26_DECK[5], *HOG_26_DECK[1:4])
    assert "does not match" in dispatcher.advance(builder.build(_battle_frame(104, hand=changed)), 0)
    builder.build(_battle_frame(120, hand=changed, ended=True))
    assert [status for _, status in builder.receipts] == ["match_ended"]
    assert dispatcher.submitted == 1 and not builder.pending


def test_rotated_slot_uses_current_balance_and_late_event_confirms_once():
    builder, controller = LiveObservationBuilder(), Mock()
    dispatcher = LivePlayDispatcher(builder, controller)
    dispatcher.submit(builder.build(_battle_frame(100, cursor=4)), 4, (_play(1),), None)
    rotated = (HOG_26_DECK[0], HOG_26_DECK[4], *HOG_26_DECK[2:4])
    prior = _event(4, 100, 1, 4)
    obs = builder.build(_battle_frame(108, hand=rotated, elixir=2, events=[prior], cursor=4))
    assert not builder.receipts and obs.action_mask.reasons["effective_elixir"] == 2
    assert obs.action_mask.hand_slots[1]
    receipt = _event(5, 105, 1, 4)
    builder.build(_battle_frame(145, hand=rotated, elixir=2, events=[receipt], cursor=5))
    assert [status for _, status in builder.receipts] == ["confirmed"]
    builder.build(_battle_frame(146, hand=rotated, elixir=2, events=[receipt], cursor=5))
    assert not builder.receipts


def test_checkpoint_follows_up_against_an_advancing_push_with_the_first_receipt_outstanding():
    import torch
    from native_runner.training.v4.policy_session import PolicySessionV4, load_policy_v4

    root = Path(__file__).resolve().parents[2]
    path = root / live_ladder.DEFAULT_CHECKPOINT
    if not path.is_file():
        pytest.skip("requires the fitted local Hog checkpoint")
    torch.set_num_threads(1)
    loaded = load_policy_v4(live_ladder._checkpoint(path), device="cpu", delay_policy="earliest")
    assert loaded.checkpoint_sha256 == live_ladder.EXPECTED_CHECKPOINT_SHA256
    builder = LiveObservationBuilder()
    dispatcher = LivePlayDispatcher(builder, Mock())
    tensorizer = build_live_tensorizer_v4(HOG_26_DECK, actor_owner=0, own_form_availability=(0,) * 8)
    session = PolicySessionV4(loaded.model, tensorizer, device="cpu", sample=False, validate_tensors=False)
    session.start_episode(builder.build(_battle_frame(0, elixir=7)), initial_elixir={0: 7, 1: 7})
    hand = list(HOG_26_DECK[:4])
    plays = []
    first_cost = 0
    rotated = False
    try:
        for tick in range(90, 311, 5):
            if plays and tick >= plays[0][0] + 20 and not rotated:
                first = plays[0][1]
                hand[first.hand_slot] = HOG_26_DECK[4]
                rotated = True
            elixir = min(10, 7 + tick / 56) - (first_cost if rotated else 0)
            raw = _battle_frame(tick, hand=hand, elixir=elixir)

            raw["objects"].append(dict(nativeObjectId=15, owner=1, cardId=26000003, dataGlobalId=34000003,
                x=14300, y=max(6900, 15400 - (tick - 90) * 50), targetX=14500, targetY=6500, hp=3800, maxHp=3800))
            raw["returned"] = len(raw["objects"])
            obs = builder.build(raw)
            if dispatcher.follow_up is not None:
                follow_up = dispatcher.follow_up[0]
                status = dispatcher.advance(obs, 0)
                if status == "submitted":
                    plays.append((tick, follow_up))
                    break
                if status == "scheduled":
                    continue
                tensorizer.retain_previous_action_prefix(1)
            decision = session.decide(obs)
            decoded = decision.decoded.actions
            if dispatcher.submit(obs, 0, decoded, 2):
                play = next(a for a in decoded if a.kind is ActionKind.PLAY_CARD)
                plays.append((tick, play))
                if len(plays) == 2:
                    break
                first_cost = builder.bundle.card_specs[play.card_id].elixir_cost
    finally:
        session.end_episode()
    assert len(plays) == 2
    assert plays[0][1].card_id != plays[1][1].card_id
    assert 0 < plays[1][0] - plays[0][0] < 155
    assert len(builder.pending) == 2 and not builder.receipts
    print("Offline checkpoint follow-up", [(tick, a.card_id, a.target_grid) for tick, a in plays])
