from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from native_runner.live_probe import live_contracts
from native_runner.live_probe.live_contracts import LiveLadderError, validate_fair_payload
from native_runner.live_probe.memory_types import ExternalMemoryError
from native_runner.live_probe.play_ledger import PlayLedger
from native_runner.live_probe.public_snapshot import PublicBattleSnapshot
from native_runner.live_probe.touch_layout import TouchLayout
from native_runner.tests.live_fixtures import battle_payload


@dataclass(frozen=True)
class CardPlay:
    card_id: int = 123
    hand_slot: int = 0
    execute_offset_ticks: int = 1


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.ledger = PlayLedger(execution_delay_ticks=2, receipt_timeout_ticks=10)
        self.action = CardPlay()
        self.ledger.record(self.action, cost=3, requested_tick=10, source_cursor=4)
        self.event = {"owner": 0, "cardId": self.action.card_id, "tick": 10, "sequence": 5}

    def test_rotation_and_event_can_arrive_in_either_order(self):
        for event_first in (True, False):
            with self.subTest(event_first=event_first):
                ledger = PlayLedger(2, 10)
                ledger.record(self.action, cost=3, requested_tick=10, source_cursor=4)
                hand = [self.action.card_id] if event_first else [999]
                ledger.settle(11, False, hand, [self.event] if event_first else [])
                self.assertEqual(ledger.receipts, [])
                ledger.settle(12, False, [999], [] if event_first else [self.event])
                self.assertEqual([status for _, status in ledger.receipts], ["confirmed"])
                self.assertEqual(ledger.pending, [])
                ledger.settle(13, False, [999], [self.event])
                self.assertEqual(ledger.receipts, [])

    def test_prior_and_opponent_events_cannot_confirm_a_rotation(self):
        ignored = [dict(self.event, sequence=4), dict(self.event, tick=9), dict(self.event, owner=1)]
        self.ledger.settle(22, False, [999], ignored)
        self.assertTrue(self.ledger.pending[0].slot_changed)
        self.assertFalse(self.ledger.pending[0].event_seen)
        self.ledger.settle(23, False, [999], [])
        self.assertEqual([status for _, status in self.ledger.receipts], ["timed_out"])

    def test_terminal_frame_settles_an_unconfirmed_play(self):
        self.ledger.settle(23, True, [self.action.card_id], [])
        self.assertEqual([status for _, status in self.ledger.receipts], ["match_ended"])


class PublicSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.frame = battle_payload()
        own = self.frame["players"][0]
        self.state = PublicBattleSnapshot({own["hand"][0]: 1})
        self.inputs = {
            "generation": 1, "tick": 0, "finalized": False, "world_result": -1,
            "account_id": own["accountId"], "deck": own["deck"], "hand": own["hand"],
            "cycle": own["cycle"], "elixir": own["elixirRaw"], "objects": self.frame["objects"],
            "after_event": 0,
        }

    def test_initial_frame_matches_the_existing_wire_format(self):
        self.assertEqual(self.state.publish(**self.inputs), self.frame)

    def test_acknowledged_events_are_excluded_from_the_next_frame(self):
        self.state.publish(**self.inputs)
        hand = self.inputs["hand"]
        rotated = [self.inputs["cycle"][0], *hand[1:]]
        frame = self.state.publish(**dict(self.inputs, tick=20, hand=rotated))
        self.assertEqual([event["cardId"] for event in frame["events"]], [hand[0]])
        following = self.state.publish(**dict(self.inputs, tick=21, hand=rotated, after_event=frame["eventCursor"]))
        self.assertEqual(following["events"], [])
        self.assertEqual(following["sequence"], 3)

    def test_a_new_battle_needs_its_own_tower_evidence(self):
        self.state.publish(**self.inputs)
        empty = dict(self.inputs, objects=[])
        self.state.publish(**empty)
        with self.assertRaisesRegex(ExternalMemoryError, "both Tower Troops"):
            PublicBattleSnapshot({}).publish(**empty)


class FairPayloadTests(unittest.TestCase):
    def setUp(self):
        clock = Mock(wraps=datetime)
        clock.now.side_effect = lambda tz: datetime(2026, 11, 28, 12, tzinfo=tz)
        frozen_time = patch.object(live_contracts, "datetime", clock)
        frozen_time.start()
        self.addCleanup(frozen_time.stop)

    def test_opponent_identity_is_rejected_even_when_null(self):
        frame = battle_payload()
        frame["players"][1]["accountId"] = None
        with self.assertRaisesRegex(LiveLadderError, "identity leaked"):
            validate_fair_payload(frame)

    def test_invalid_card_types_get_a_contract_error(self):
        frame = battle_payload()
        frame["players"][0]["hand"][0] = []
        with self.assertRaisesRegex(LiveLadderError, "live Hog 2.6 hand"):
            validate_fair_payload(frame)

    def test_normalization_leaves_the_source_envelope_intact(self):
        frame = battle_payload()
        validated = validate_fair_payload(frame)
        self.assertEqual(validated["localAccountId"], 123)
        self.assertNotIn("localAccountId", frame)
        self.assertIsNot(validated["players"][0], frame["players"][0])


class TouchLayoutTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "live_probe" / "mumu_900x1600.json"
        self.layout = TouchLayout.load(path)

    def test_screen_orientation_and_subcell_offsets(self):
        normal = self.layout.grid_pixel((0, 0))
        flipped = replace(self.layout, owner_zero_at_screen_bottom=False).grid_pixel((0, 0))
        self.assertEqual(normal[0], flipped[0])
        self.assertGreater(normal[1], flipped[1])
        offset = self.layout.grid_pixel((0, 0), (0.25, 0.25))
        self.assertGreater(offset[0], normal[0])
        self.assertLess(offset[1], normal[1])

    def test_direct_construction_validates_layout_bounds(self):
        with self.assertRaises(LiveLadderError):
            replace(self.layout, arena_left=self.layout.arena_right)
