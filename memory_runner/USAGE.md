# Run the Hog cycle memory policy

The runner reads the local hand, cycle, elixir, clock, and visible units from the attested Null's Royale process. It uses a single CPU inference thread. The supplied touch layout is for a 900 × 1600 portrait MuMu display.

## Local setup

Use Python 3.13 and a FirstLight checkout with its native simulation data and production semantic catalog installed. Pass its root with `--firstlight`. The bundled reader verifies the package, ARM64 build, and authorization window specified in `native_runner/live_probe/game_profile.py`. The emulator needs its existing root session enabled.

Run these commands from the NullsAI repository root in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
.\.venv\Scripts\python -m pip install -r memory_runner/requirements.txt
git clone --filter=blob:none --no-checkout https://github.com/cochon123/clash-royale-ai .runtime/reference
git -C .runtime/reference sparse-checkout set src engine/rl engine/data data
git -C .runtime/reference checkout --detach 5609f61a10621ea5afb8669dfb787d6c6fde3cfe
```

Place the v4.4.2 base checkpoint `best_model.pt`, the fitted residual `selected.pt`, and the replay placement table `tile-prior.npz` in `.runtime/models/`, or specify their paths with `--base`, `--checkpoint`, and `--tile-prior`. Both checkpoints are verified by SHA-256. To regenerate the placement table from an existing IL_Replay cache containing `manifest.json` and the split NPZ shards, run `python memory_runner/fit_tile_prior.py PATH_TO_CACHE .runtime/models/tile-prior.npz`.

## Play one match

Open the game with the Hog 2.6 deck selected. Use a fresh log filename for each run.

```powershell
$device = @('--firstlight', 'D:/FirstLight', '--player-tag', 'YOUR_PLAYER_TAG', '--serial', '127.0.0.1:16416')
.\.venv\Scripts\python memory_runner/run.py @device --log .runtime/match-01.jsonl
```

Start the match in-game when the terminal prints `ready`. The runner waits for a moving battle clock and verifies the local account and deck before playing. The `armed` event reports attachment time in milliseconds. The first observation reports the time from that lookup to a usable board. Lobby waiting is reported separately. Decisions record the observed threat, its position, the intended placement, and inference time. A confirmed receipt means the game accepted the play.

Add `--observe-only` to read the match without loading the policy or submitting plays. Use `--adb` to select an ADB executable. Create `.runtime/match-01.STOP` to stop this run.
