from __future__ import annotations

import hashlib
from pathlib import Path

from ..battle_env import TOWER_LAYOUT
from .game_profile import (
    AUTHORIZATION_END_EPOCH_EXCLUSIVE,
    DEFAULT_READER_PORT,
    EXPECTED_LIBG_BUILD_ID,
    EXPECTED_LIBG_SHA256,
    EXPECTED_PACKAGE,
    EXPECTED_PROTOCOL,
    EXPECTED_READER_SHA256,
    EXPECTED_SCHEMA,
    EXPECTED_STATUS_SCHEMA,
    EXPECTED_VERSION,
    HOG_26_DECK,
    HOG_26_SET,
    SUPPORTED_TOWER_TROOPS,
)
from .live_contracts import AUTHORIZATION_END_EXCLUSIVE, AUTHORIZATION_START, PARIS

EXPECTED_CHECKPOINT_SHA256 = "fc7bb42e4757155242498239f90031055973838443481b3c6c7e08e04d4ba390"
DEFAULT_CHECKPOINT = Path("checkpoints/Everlight-Hog26/hog26-push-defense-selected.pt")
PREVIOUS_CHECKPOINT_SHA256 = "4af3fef5fdc7ade446e627cf6ebf0c87eb74e65e66f985bb11820135f8e5d2bb"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = DEFAULT_READER_PORT
TICK_MS = 50
RECEIPT_TIMEOUT_TICKS = 200
RULESET_ID = hashlib.sha256(b"nullsroyale-live-16.635.1-fair.v1").hexdigest()
EPISODE_PREFIX = "nullsroyale-live"
TOWER_POSITION = {(owner, x, y): (entity_id, kind) for entity_id, owner, kind, x, y in TOWER_LAYOUT}
