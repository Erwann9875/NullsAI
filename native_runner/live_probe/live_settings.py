from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..battle_env import TOWER_LAYOUT

PARIS = ZoneInfo("Europe/Paris")
AUTHORIZATION_START = datetime(2026, 11, 23, 0, 0, tzinfo=PARIS)
AUTHORIZATION_END_EXCLUSIVE = datetime(2026, 12, 2, 0, 0, tzinfo=PARIS)
AUTHORIZATION_END_EPOCH_EXCLUSIVE = 1_796_166_000

EXPECTED_SCHEMA = "nullsroyale-fair-live.v1"
EXPECTED_STATUS_SCHEMA = "nullsroyale-external-status.v1"
EXPECTED_PROTOCOL = "nullsroyale-external-control.v1"
EXPECTED_PACKAGE = "nullsroyale.rel.free"
EXPECTED_VERSION = "16.635.1"
EXPECTED_READER_SHA256 = "01a6f9b96214927de0d52a51ee67444395eba412ab1c8c718634d2317c76cdd9"
DEFAULT_READER_PORT = 26_792
EXPECTED_LIBG_SHA256 = "0b580222e5a3502cf153bb5e6297ec32d2b154b605b777c74873dd75722e4eaa"
EXPECTED_LIBG_BUILD_ID = "179ac4bd38c205bc5041d372a79f1c2f929b1c30"
EXPECTED_CHECKPOINT_SHA256 = "fc7bb42e4757155242498239f90031055973838443481b3c6c7e08e04d4ba390"
DEFAULT_CHECKPOINT = Path("checkpoints/Everlight-Hog26/hog26-push-defense-selected.pt")
PREVIOUS_CHECKPOINT_SHA256 = "4af3fef5fdc7ade446e627cf6ebf0c87eb74e65e66f985bb11820135f8e5d2bb"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = DEFAULT_READER_PORT
TICK_MS = 50
RECEIPT_TIMEOUT_TICKS = 200
RULESET_ID = hashlib.sha256(b"nullsroyale-live-16.635.1-fair.v1").hexdigest()
EPISODE_PREFIX = "nullsroyale-live"

HOG_26_DECK = (
    26_000_010,
    26_000_014,
    26_000_021,
    26_000_030,
    26_000_038,
    27_000_000,
    28_000_000,
    28_000_011,
)
HOG_26_SET = frozenset(HOG_26_DECK)
SUPPORTED_TOWER_TROOPS = frozenset({159_000_000, 159_000_001, 159_000_002, 159_000_004})
TOWER_POSITION = {(owner, x, y): (entity_id, kind) for entity_id, owner, kind, x, y in TOWER_LAYOUT}
