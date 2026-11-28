from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ADB = Path(shutil.which('adb') or 'C:/Program Files/Netease/MuMuPlayer/nx_main/adb.exe')


def configure(firstlight, reference=None):
    if not (firstlight / 'native_runner' / 'battle_env.py').is_file():
        raise ValueError('--firstlight must point to the FirstLight checkout containing native_runner/battle_env.py')
    sys.path.insert(1, str(firstlight.resolve()))
    import native_runner
    native_runner.__path__.insert(0, str(REPO / 'native_runner'))
    if reference is not None:
        if not (reference / 'engine' / 'rl' / 'banana_policy.py').is_file():
            raise ValueError('--reference must point to the upstream clash-royale-ai checkout')
        sys.path[1:1] = [str(reference.resolve() / 'src'), str(reference.resolve() / 'engine')]


def account_id(tag):
    alphabet = '0289PYLQGRJCUV'
    value = 0
    for character in tag.removeprefix('#').upper().replace('O', '0'):
        value = value * len(alphabet) + alphabet.index(character)
    return ((value % 256) << 32) | (value // 256)


@contextmanager
def bridge(args, costs):
    from native_runner.live_probe.external_source import AdbExternalReaderRuntime, ExternalLiveClient
    from native_runner.live_probe.live_contracts import require_authorization, validate_status
    require_authorization()

    class RootShellRuntime(AdbExternalReaderRuntime):
        forwarded = False
        uploaded = False

        def _forward(self):
            self._run('forward', '--no-rebind', f'tcp:{self.port}', f'tcp:{self.port}')
            self.forwarded = True

        def _deploy_and_start(self, pid, reader_hash):
            self._run('push', str(self.reader_binary), self.remote_reader)
            self.uploaded = True
            self._run('shell', 'chmod', '700', self.remote_reader)
            if self._run('shell', 'sha256sum', self.remote_reader).split()[0] != reader_hash:
                raise RuntimeError('The device reader does not match the pinned binary')
            command = f'exec {self.remote_reader} {pid} {self.port}'
            self.process = subprocess.Popen(
                self.command + ['shell', '-T', f'su -c {shlex.quote(command)}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            self.owns_process = True
            for _ in range(30):
                if self.process.poll() is not None:
                    break
                if self._matching_reader(pid):
                    return
                time.sleep(.1)
            raise RuntimeError('The read-only sidecar did not start with the existing root session')

    runtime = RootShellRuntime(
        adb=args.adb, serial=args.serial, port=args.port,
        reader_binary=REPO/'native_runner/live_probe/out/nullsroyale-memory-reader',
        remote_reader=f'/data/local/tmp/nullsai-match-{args.port}-{os.getpid()}',
    )
    client = None
    try:
        runtime.start()
        client = ExternalLiveClient(runtime, local_account_id=account_id(args.player_tag), card_costs=costs)
        validate_status(client.status())
        yield client
    finally:
        if client is not None:
            client.disarm()
        runtime.close()
        if runtime.forwarded:
            runtime._run('forward', '--remove', f'tcp:{args.port}', check=False)
        if runtime.uploaded:
            runtime._run('shell', 'rm', '-f', runtime.remote_reader, check=False)
