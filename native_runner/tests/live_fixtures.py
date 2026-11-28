from __future__ import annotations

from native_runner.live_probe.game_profile import AUTHORIZATION_END_EPOCH_EXCLUSIVE, HOG_26_DECK


def battle_payload() -> dict[str, object]:
    positions = (
        (0, 9000, 3000),
        (0, 3500, 6500),
        (0, 14500, 6500),
        (1, 9000, 29000),
        (1, 3500, 25500),
        (1, 14500, 25500),
    )
    objects = [
        {
            "nativeObjectId": index,
            "owner": owner,
            "cardId": -1,
            "dataGlobalId": 35_000_000 if x == 9000 else 35_000_001,
            "x": x,
            "y": y,
            "targetX": x,
            "targetY": y,
            "hp": 3000,
            "maxHp": 3000,
        }
        for index, (owner, x, y) in enumerate(positions, 1)
    ]
    return {
        "ok": True,
        "schema": "nullsroyale-fair-live.v1",
        "generation": 1,
        "sequence": 1,
        "tick": 0,
        "ended": False,
        "finalized": False,
        "worldResultRaw": -1,
        "localOwner": 0,
        "authorizationEndUtcExclusive": AUTHORIZATION_END_EPOCH_EXCLUSIVE,
        "players": [
            {
                "owner": 0,
                "accountId": 123,
                "crownsRaw": 0,
                "elixirRaw": 50_000,
                "elixir": 5.0,
                "towerTroopId": 159_000_000,
                "privateStateVisible": True,
                "hand": list(HOG_26_DECK[:4]),
                "cycle": list(HOG_26_DECK[4:]),
                "nextCard": HOG_26_DECK[4],
                "deck": list(HOG_26_DECK),
            },
            {
                "owner": 1,
                "crownsRaw": 0,
                "towerTroopId": 159_000_000,
                "elixirRaw": None,
                "elixir": None,
                "privateStateVisible": False,
                "hand": [],
                "cycle": [],
                "nextCard": None,
                "deck": [],
            },
        ],
        "objects": objects,
        "returned": len(objects),
        "objectsTruncated": False,
        "eventsTruncated": False,
        "events": [],
        "eventCursor": 0,
    }
