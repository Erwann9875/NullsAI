from __future__ import annotations

import json
import socket
from typing import Protocol, Sequence

from .memory_types import (
    AUTHORIZATION_END_UTC_EXCLUSIVE,
    MAX_TRANSFER,
    PACKAGE,
    TOKEN,
    ExternalMemoryError,
    ProcessMapping,
    parse_maps,
)

def _recv_exact(stream: object, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = stream.read(size - len(output))
        if not chunk:
            raise ExternalMemoryError("external reader closed during a binary response")
        output.extend(chunk)
    return bytes(output)

class MemoryTransport(Protocol):
    def status(self) -> dict[str, object]: ...

    def maps(self) -> tuple[ProcessMapping, ...]: ...

    def scan_game_states(self) -> tuple[tuple[int, int], ...]: ...

    def trophy_count(self) -> int | None: ...

    def read(self, address: int, size: int) -> bytes: ...

    def readv(self, ranges: Sequence[tuple[int, int]]) -> tuple[bytes, ...]: ...

class ExternalReaderClient:

    def __init__(self, host: str = "127.0.0.1", port: int = 26_792, *, timeout: float = 10.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def _binary(self, command: str) -> bytes:
        if "\n" in command or "\r" in command:
            raise ValueError("reader command must be one line")
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as connection:
            connection.settimeout(self.timeout)
            stream = connection.makefile("rb")
            connection.sendall(command.encode("ascii") + b"\n")
            header = stream.readline(4096)
            if not header:
                raise ExternalMemoryError("external reader returned no response")
            line = header.decode("utf-8", errors="replace").strip()
            if line.startswith("ERR "):
                raise ExternalMemoryError(line[4:])
            fields = line.split()
            if len(fields) != 2 or fields[0] != "OK" or not fields[1].isdigit():
                raise ExternalMemoryError(f"invalid external reader header: {line!r}")
            size = int(fields[1])
            if size < 0 or size > MAX_TRANSFER:
                raise ExternalMemoryError("external reader response exceeds the fixed host bound")
            return _recv_exact(stream, size)

    def status(self) -> dict[str, object]:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as connection:
            connection.settimeout(self.timeout)
            stream = connection.makefile("rb")
            connection.sendall(f"status {TOKEN}\n".encode("ascii"))
            line = stream.readline(4096).decode("utf-8", errors="replace").strip()
        if not line.startswith("OK "):
            raise ExternalMemoryError(line.removeprefix("ERR ") or "external reader status failed")
        try:
            value = json.loads(line[3:])
        except json.JSONDecodeError as error:
            raise ExternalMemoryError("external reader returned invalid status JSON") from error
        expected = {
            "schema": "nullsroyale-external-reader.v1",
            "package": PACKAGE,
            "readOnly": True,
            "authorized": True,
            "processMatches": True,
            "authorizationEndUtcExclusive": AUTHORIZATION_END_UTC_EXCLUSIVE,
        }
        if not isinstance(value, dict) or any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise ExternalMemoryError(f"external reader attestation failed: {value!r}")
        return value

    def maps(self) -> tuple[ProcessMapping, ...]:
        return parse_maps(self._binary(f"maps {TOKEN}"))

    def scan_game_states(self) -> tuple[tuple[int, int], ...]:
        payload = self._binary(f"scan-gamestate {TOKEN}")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ExternalMemoryError("external reader returned an invalid GameState scan") from error
        if not isinstance(value, dict) or value.get("schema") != "nullsroyale-gamestate-scan.v1":
            raise ExternalMemoryError("external reader returned an unsupported GameState scan")
        candidates = value.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > 4_096:
            raise ExternalMemoryError("external GameState scan exceeded its candidate bound")
        result: list[tuple[int, int]] = []
        for candidate in candidates:
            if (
                not isinstance(candidate, list)
                or len(candidate) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in candidate)
            ):
                raise ExternalMemoryError("external reader returned a malformed GameState candidate")
            result.append((candidate[0], candidate[1]))
        return tuple(result)

    def trophy_count(self) -> int | None:
        """Read the local avatar's trophies, or None while the avatar loads."""
        payload = self._binary(f"trophies {TOKEN}")
        if len(payload) != 4:
            raise ExternalMemoryError("external reader returned an invalid trophy payload length")
        count = int.from_bytes(payload, "little", signed=True)
        if count < -1:
            raise ExternalMemoryError("external reader returned an invalid trophy count")
        return None if count == -1 else count

    def read(self, address: int, size: int) -> bytes:
        if address <= 0 or size <= 0 or size > MAX_TRANSFER:
            raise ValueError("invalid external memory read")
        return self._binary(f"read {TOKEN} {address:x} {size}")

    def readv(self, ranges: Sequence[tuple[int, int]]) -> tuple[bytes, ...]:
        if not ranges:
            return ()
        if len(ranges) > 1_024 or sum(size for _, size in ranges) > MAX_TRANSFER:
            raise ValueError("external readv exceeds fixed protocol limits")
        arguments = " ".join(f"{address:x}:{size}" for address, size in ranges)
        payload = self._binary(f"readv {TOKEN} {arguments}")
        output: list[bytes] = []
        cursor = 0
        for _, size in ranges:
            output.append(payload[cursor : cursor + size])
            cursor += size
        if cursor != len(payload):
            raise ExternalMemoryError("external readv length mismatch")
        return tuple(output)
