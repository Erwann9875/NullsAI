from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Mapping, TypeGuard
from zoneinfo import ZoneInfo

from .game_profile import (
    AUTHORIZATION_END_EPOCH_EXCLUSIVE,
    EXPECTED_LIBG_BUILD_ID,
    EXPECTED_LIBG_SHA256,
    EXPECTED_PACKAGE,
    EXPECTED_PROTOCOL,
    EXPECTED_READER_SHA256,
    EXPECTED_SCHEMA,
    EXPECTED_STATUS_SCHEMA,
    EXPECTED_VERSION,
    HOG_26_SET,
    SUPPORTED_TOWER_TROOPS,
)


PARIS = ZoneInfo("Europe/Paris")
AUTHORIZATION_START = datetime(2026, 11, 23, 0, 0, tzinfo=PARIS)
AUTHORIZATION_END_EXCLUSIVE = datetime(2026, 12, 2, 0, 0, tzinfo=PARIS)


class LiveLadderError(RuntimeError):
    pass


def authorization_open(now: datetime | None = None) -> bool:
    actual = datetime.now(tz=PARIS) if now is None else now.astimezone(PARIS)
    return AUTHORIZATION_START <= actual < AUTHORIZATION_END_EXCLUSIVE


def require_authorization(now: datetime | None = None) -> None:
    if not authorization_open(now):
        raise LiveLadderError(
            "ladder control is outside the authorized 2026-11-23 through 2026-12-01 Europe/Paris window"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_int(value: object, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveLadderError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise LiveLadderError(f"{label} is below {minimum}")
    if maximum is not None and value > maximum:
        raise LiveLadderError(f"{label} exceeds {maximum}")
    return value


def validate_status(value: Mapping[str, object]) -> dict[str, object]:
    status = dict(value)
    expected = {
        "ok": True,
        "schema": EXPECTED_STATUS_SCHEMA,
        "protocol": EXPECTED_PROTOCOL,
        "package": EXPECTED_PACKAGE,
        "gameVersion": EXPECTED_VERSION,
        "primaryCpuAbi": "arm64-v8a",
        "readOnly": True,
        "injected": False,
        "libgSha256": EXPECTED_LIBG_SHA256,
        "libgExpectedSha256": EXPECTED_LIBG_SHA256,
        "libgBuildId": EXPECTED_LIBG_BUILD_ID,
        "readerSha256": EXPECTED_READER_SHA256,
        "localOwner": 0,
    }
    for key, expected_value in expected.items():
        actual = status.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise LiveLadderError(f"probe status {key} is {actual!r}, expected {expected_value!r}")
    if status.get("authorizationEndUtcExclusive") != AUTHORIZATION_END_EPOCH_EXCLUSIVE:
        raise LiveLadderError("external reader authorization expiry differs from the host gate")
    require_authorization()
    return status


def _validate_envelope(raw: Mapping[str, object]) -> None:
    if raw.get("ok") is not True or raw.get("schema") != EXPECTED_SCHEMA:
        raise LiveLadderError("probe returned an unsupported live observation")
    if raw.get("localOwner") != 0:
        raise LiveLadderError("only the locally attested owner-0 perspective is supported")
    if raw.get("authorizationEndUtcExclusive") != AUTHORIZATION_END_EPOCH_EXCLUSIVE:
        raise LiveLadderError("observation expiry differs from the authorization gate")
    require_authorization()
    _exact_int(raw.get("generation"), "generation", minimum=1)
    _exact_int(raw.get("sequence"), "sequence", minimum=1)
    _exact_int(raw.get("tick"), "tick", minimum=0)
    if raw.get("objectsTruncated") is not False or raw.get("eventsTruncated") is not False:
        raise LiveLadderError("probe truncated public state; control is disabled")


def _validate_redaction(enemy: Mapping[str, object]) -> None:
    if enemy.get("owner") != 1 or enemy.get("privateStateVisible") is not False:
        raise LiveLadderError("opponent row is not explicitly redacted")
    if "accountId" in enemy:
        raise LiveLadderError("opponent account identity leaked into the FAIR payload")
    redacted = {"elixirRaw": None, "elixir": None, "hand": [], "cycle": [], "nextCard": None, "deck": []}
    for key, expected in redacted.items():
        if enemy.get(key) != expected:
            raise LiveLadderError(f"opponent-private field {key} is not redacted")


def _known_cards(cards: object) -> TypeGuard[list[int]]:
    return isinstance(cards, list) and all(type(card) is int and card in HOG_26_SET for card in cards)


def _validate_local_player(own: Mapping[str, object]) -> int:
    if own.get("owner") != 0 or own.get("privateStateVisible") is not True:
        raise LiveLadderError("local private state is missing")
    account_id = _exact_int(own.get("accountId"), "local account ID", minimum=1)
    _exact_int(own.get("elixirRaw"), "own elixirRaw", minimum=0, maximum=100_000)
    deck = own.get("deck")
    hand = own.get("hand")
    cycle = own.get("cycle")
    if not _known_cards(deck) or len(deck) != 8 or frozenset(deck) != HOG_26_SET:
        raise LiveLadderError("the active local deck is not the attested Hog 2.6 deck")
    if not _known_cards(hand) or len(hand) != 4 or len(set(hand)) != 4:
        raise LiveLadderError("the live Hog 2.6 hand is incomplete or invalid")
    if not _known_cards(cycle):
        raise LiveLadderError("the local cycle contains an unexpected card")
    return account_id


def _players(raw: Mapping[str, object]) -> tuple[list[dict[str, object]], int]:
    players = raw.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise LiveLadderError("live observation must contain two player rows")
    own, enemy = players
    if not isinstance(own, Mapping) or not isinstance(enemy, Mapping):
        raise LiveLadderError("player rows must be objects")
    account_id = _validate_local_player(own)
    _validate_redaction(enemy)
    for index, player in enumerate((own, enemy)):
        tower_troop = _exact_int(player.get("towerTroopId"), f"players[{index}].towerTroopId")
        if tower_troop not in SUPPORTED_TOWER_TROOPS:
            raise LiveLadderError("live Tower Troop is outside the model's competitive vocabulary")
    return [dict(own), dict(enemy)], account_id


def _public_objects(raw: Mapping[str, object]) -> list[dict[str, object]]:
    objects = raw.get("objects")
    if not isinstance(objects, list) or raw.get("returned") != len(objects):
        raise LiveLadderError("public object count is inconsistent")
    identities: set[int] = set()
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise LiveLadderError(f"objects[{index}] is not an object")
        native_id = _exact_int(item.get("nativeObjectId"), f"objects[{index}].nativeObjectId", minimum=1)
        if native_id in identities:
            raise LiveLadderError("duplicate native object identity")
        identities.add(native_id)
        _exact_int(item.get("owner"), f"objects[{index}].owner", minimum=-1, maximum=1)
        _exact_int(item.get("cardId"), f"objects[{index}].cardId")
        _exact_int(item.get("dataGlobalId"), f"objects[{index}].dataGlobalId", minimum=1, maximum=0xFFFFFFFF)
        _exact_int(item.get("x"), f"objects[{index}].x")
        _exact_int(item.get("y"), f"objects[{index}].y")
        if (item.get("hp") is None) != (item.get("maxHp") is None):
            raise LiveLadderError("object HP and max HP availability disagree")
        if item.get("hp") is not None:
            hp = _exact_int(item.get("hp"), f"objects[{index}].hp", minimum=0)
            maximum = _exact_int(item.get("maxHp"), f"objects[{index}].maxHp", minimum=1)
            if hp > maximum:
                raise LiveLadderError("object HP exceeds max HP")
    return [dict(item) for item in objects]


def _card_events(raw: Mapping[str, object]) -> list[dict[str, object]]:
    events = raw.get("events")
    if not isinstance(events, list):
        raise LiveLadderError("public card event list is missing")
    last_sequence = -1
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("kind") != "card_play":
            raise LiveLadderError(f"events[{index}] is not a public card play")
        sequence = _exact_int(event.get("sequence"), f"events[{index}].sequence", minimum=1)
        if sequence <= last_sequence:
            raise LiveLadderError("public card events are not strictly ordered")
        last_sequence = sequence
        _exact_int(event.get("owner"), f"events[{index}].owner", minimum=0, maximum=1)
        _exact_int(event.get("cardId"), f"events[{index}].cardId", minimum=1)
        _exact_int(event.get("cost"), f"events[{index}].cost", minimum=0, maximum=15)
        _exact_int(event.get("formCode"), f"events[{index}].formCode", minimum=0, maximum=3)
    return [dict(event) for event in events]


def validate_fair_payload(value: Mapping[str, object]) -> dict[str, object]:
    raw = dict(value)
    _validate_envelope(raw)
    raw["players"], raw["localAccountId"] = _players(raw)
    raw["objects"] = _public_objects(raw)
    raw["events"] = _card_events(raw)
    return raw
