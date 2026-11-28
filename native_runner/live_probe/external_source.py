from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from tempfile import NamedTemporaryFile
from typing import Callable, Mapping

from .live_contracts import LiveLadderError, sha256_file
from .game_profile import (
    DEFAULT_READER_PORT,
    EXPECTED_PROTOCOL,
    EXPECTED_READER_SHA256,
    EXPECTED_STATUS_SCHEMA,
    EXPECTED_VERSION,
)
from .external_memory import (
    AUTHORIZATION_END_UTC_EXCLUSIVE,
    EXPECTED_LIBG_BUILD_ID,
    EXPECTED_LIBG_SHA256,
    HOG_26_SET,
    PACKAGE,
    BattleLayout,
    ExternalBattleReader,
    ExternalMemoryError,
    ExternalReaderClient,
    MemoryTransport,
)


EXPECTED_GAME_VERSION = EXPECTED_VERSION
DEFAULT_REMOTE_READER = "/data/local/tmp/nullsroyale-memory-reader"
LIBG_PATH_PATTERN = re.compile(r"^/data/app/[A-Za-z0-9._=+~/-]+/lib/arm64/libg\.so$")


class ExternalSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalAttestation:
    pid: int
    version: str
    primary_abi: str
    libg_path: str
    libg_sha256: str
    reader_sha256: str


class AdbExternalReaderRuntime:
    def __init__(
        self,
        *,
        adb: Path,
        serial: str,
        reader_binary: Path,
        host: str = "127.0.0.1",
        port: int = DEFAULT_READER_PORT,
        remote_reader: str = DEFAULT_REMOTE_READER,
        client_factory: Callable[..., MemoryTransport] | None = None,
    ) -> None:
        self.adb = adb.resolve()
        self.serial = serial
        self.reader_binary = reader_binary.resolve()
        self.host = host
        self.port = int(port)
        self.remote_reader = remote_reader
        self.command = [str(self.adb), "-s", self.serial]
        transport_factory = ExternalReaderClient if client_factory is None else client_factory
        self.client = transport_factory(self.host, self.port, timeout=30.0)
        self.process: subprocess.Popen[str] | None = None
        self.owns_process = False
        self.attestation: ExternalAttestation | None = None

    def _run(self, *args: str, timeout: float = 20.0, check: bool = True) -> str:
        result = subprocess.run(
            self.command + list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = (result.stdout + result.stderr).strip()
        if check and result.returncode:
            raise ExternalSourceError(output or f"ADB exited {result.returncode}")
        return output

    def _pid(self) -> int:
        output = self._run("shell", "pidof", "-s", PACKAGE)
        if not output.isdigit() or int(output) <= 0:
            raise ExternalSourceError(f"could not resolve the exact {PACKAGE} PID: {output!r}")
        return int(output)

    def _package(self) -> tuple[str, str]:
        output = self._run("shell", "dumpsys", "package", PACKAGE)
        version = re.search(r"(?m)^\s*versionName=(\S+)\s*$", output)
        abi = re.search(r"(?m)^\s*primaryCpuAbi=(\S+)\s*$", output)
        if version is None or abi is None:
            raise ExternalSourceError("package manager omitted versionName or primaryCpuAbi")
        if version.group(1) != EXPECTED_GAME_VERSION or abi.group(1) != "arm64-v8a":
            raise ExternalSourceError(
                f"unexpected game package: version={version.group(1)!r} primaryCpuAbi={abi.group(1)!r}"
            )
        return version.group(1), abi.group(1)

    def _reader_hash(self) -> str:
        if not self.reader_binary.is_file():
            raise ExternalSourceError(f"reader binary is missing: {self.reader_binary}")
        digest = sha256_file(self.reader_binary)
        if digest != EXPECTED_READER_SHA256:
            raise ExternalSourceError(f"reader SHA-256 is {digest}, expected {EXPECTED_READER_SHA256}")
        return digest

    def _forward(self) -> None:
        self._run("forward", f"tcp:{self.port}", f"tcp:{self.port}")

    def _matching_reader(self, pid: int) -> bool:
        try:
            status = self.client.status()
        except (OSError, TimeoutError, ExternalMemoryError):
            return False
        return int(status.get("pid", -1)) == pid

    def _deploy_and_start(self, pid: int, reader_hash: str) -> None:
        self._run("shell", "pkill", "-f", f"^{self.remote_reader}( |$)", check=False)
        self._run("push", str(self.reader_binary), self.remote_reader, timeout=60.0)
        self._run("shell", "chmod", "700", self.remote_reader)
        remote_hash = self._run("shell", "sha256sum", self.remote_reader).split(maxsplit=1)[0]
        if remote_hash != reader_hash:
            raise ExternalSourceError(f"remote reader SHA-256 is {remote_hash}, expected {reader_hash}")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            self.command + ["shell", self.remote_reader, str(pid), str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=flags,
        )
        self.owns_process = True
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = self.process.stdout.read().strip() if self.process.stdout is not None else ""
                raise ExternalSourceError(output or "external reader exited during startup")
            try:
                if self._matching_reader(pid):
                    return
            except (ConnectionError, socket.timeout):
                pass
            time.sleep(0.1)
        raise ExternalSourceError("external reader did not attest within eight seconds")

    def _libg(self) -> tuple[str, str]:
        maps = self.client.maps()
        paths = {mapping.name for mapping in maps if mapping.name.endswith("/lib/arm64/libg.so")}
        if len(paths) != 1:
            raise ExternalSourceError(f"expected one mapped ARM64 libg path, found {sorted(paths)}")
        path = paths.pop()
        if LIBG_PATH_PATTERN.fullmatch(path) is None:
            raise ExternalSourceError(f"mapped libg path failed validation: {path!r}")
        digest = self._run("shell", "sha256sum", path).split(maxsplit=1)[0]
        if digest != EXPECTED_LIBG_SHA256:
            raise ExternalSourceError(f"libg SHA-256 is {digest}, expected {EXPECTED_LIBG_SHA256}")
        return path, digest

    def start(self) -> ExternalAttestation:
        if not self.adb.is_file():
            raise ExternalSourceError(f"ADB does not exist: {self.adb}")
        reader_hash = self._reader_hash()
        version, abi = self._package()
        pid = self._pid()
        self._forward()
        if not self._matching_reader(pid):
            self._deploy_and_start(pid, reader_hash)
        libg_path, libg_hash = self._libg()
        self.attestation = ExternalAttestation(
            pid=pid,
            version=version,
            primary_abi=abi,
            libg_path=libg_path,
            libg_sha256=libg_hash,
            reader_sha256=reader_hash,
        )
        return self.attestation

    def ensure_current(self) -> ExternalAttestation:
        if self.attestation is None:
            raise ExternalSourceError("external reader runtime is not started")
        current = self._pid()
        if current != self.attestation.pid:
            raise ExternalSourceError(
                f"game PID changed from {self.attestation.pid} to {current}; restart Null's Royale to re-attest"
            )
        status = self.client.status()
        if int(status.get("pid", -1)) != current:
            raise ExternalSourceError("external reader PID no longer matches the game")
        return self.attestation

    def close(self) -> None:
        if self.owns_process and self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        self.process = None
        self.owns_process = False

    def __enter__(self) -> "AdbExternalReaderRuntime":
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


class ExternalLiveClient:
    def __init__(
        self,
        runtime: AdbExternalReaderRuntime,
        *,
        local_account_id: int,
        card_costs: Mapping[int, int],
    ) -> None:
        if runtime.attestation is None:
            raise ExternalSourceError("runtime must be started before creating the live client")
        self.runtime = runtime
        self.local_account_id = int(local_account_id)
        self.reader = ExternalBattleReader(
            runtime.client,
            card_costs=card_costs,
            local_account_id=self.local_account_id,
        )
        self._hint_file = Path(__file__).resolve().parents[2] / ".runtime" / "battle-clock.json"
        self._hint_scope = [runtime.serial, runtime.attestation.pid, runtime.attestation.libg_sha256,
                            self.reader.libg_base, self.local_account_id]
        try:
            scope, manager, world = json.loads(self._hint_file.read_text(encoding="utf-8"))
            if scope == self._hint_scope and all(type(n) is int and 0 < n < 2**64 for n in (manager, world)):
                self.reader._clock_hint = (manager, world)
        except (OSError, ValueError, TypeError):
            pass

    def status(self) -> dict[str, object]:
        attestation = self.runtime.ensure_current()
        layout = self.reader.layout
        return {
            "ok": True,
            "schema": EXPECTED_STATUS_SCHEMA,
            "protocol": EXPECTED_PROTOCOL,
            "package": PACKAGE,
            "gameVersion": attestation.version,
            "primaryCpuAbi": attestation.primary_abi,
            "pid": attestation.pid,
            "readOnly": True,
            "injected": False,
            "libgSha256": attestation.libg_sha256,
            "libgExpectedSha256": EXPECTED_LIBG_SHA256,
            "libgBuildId": EXPECTED_LIBG_BUILD_ID,
            "readerSha256": attestation.reader_sha256,
            "authorizationEndUtcExclusive": AUTHORIZATION_END_UTC_EXCLUSIVE,
            "localOwner": 0,
            "snapshotReady": layout is not None,
            "hog26DeckMatch": layout is not None and frozenset(layout.deck) == HOG_26_SET,
            "localAccountId": self.local_account_id,
        }

    def arm(self, account_id: int, *, quick: bool = False) -> BattleLayout:
        self.runtime.ensure_current()
        if int(account_id) != self.local_account_id:
            raise ExternalSourceError("requested account does not match the configured local account")
        layout = self.reader.discover(quick=quick)
        try:
            self._hint_file.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._hint_file.parent,
                                    prefix=".battle-clock-", delete=False) as output:
                json.dump([self._hint_scope, layout.manager, layout.world], output)
            try:
                os.replace(output.name, self._hint_file)
            finally:
                Path(output.name).unlink(missing_ok=True)
        except OSError:
            pass
        return layout

    def disarm(self) -> None:
        self.reader.layout = None

    def observe(self, event_cursor: int) -> dict[str, object]:
        self.runtime.ensure_current()
        if self.reader.layout is None:
            raise ExternalSourceError("external live client is not armed")
        return self.reader.snapshot(int(event_cursor))


def _observe_with_snapshot_retry(
    client: ExternalLiveClient,
    event_cursor: int,
    *,
    attempts: int = 40,
) -> dict[str, object]:
    if attempts < 1:
        raise ValueError("snapshot attempts must be positive")
    last_error: ExternalMemoryError | None = None
    for _attempt in range(attempts):
        try:
            return client.observe(event_cursor)
        except ExternalMemoryError as error:
            last_error = error
            time.sleep(0.01)
    raise LiveLadderError(
        f"live memory stayed inconsistent across {attempts} consecutive snapshot retries"
    ) from last_error
