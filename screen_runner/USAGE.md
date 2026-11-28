# Run the screen-based Hog cycle policy

Use Python 3.13, a 900 × 1600 portrait MuMu display, and the eight-card Hog cycle deck listed in `live_decision.py`. The runner reads the visible hand, elixir bar, and arena, sends a drag for each play, and confirms it when the card in that hand slot changes. It uses one PyTorch CPU thread. On Windows it also lowers the process priority.

## Local dependencies

From the NullsAI repository root, install the CPU dependencies into a virtual environment and check out the upstream policy code at the revision used for the match.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
.\.venv\Scripts\python -m pip install -r screen_runner/requirements.txt
git clone --filter=blob:none --no-checkout https://github.com/cochon123/clash-royale-ai .runtime/reference
git -C .runtime/reference sparse-checkout set src engine/rl engine/data data
git -C .runtime/reference checkout --detach 5609f61a10621ea5afb8669dfb787d6c6fde3cfe
```

Supply your local v4.4.2 `best_model.pt` and v3 `selected.pt`. The runner checks each file’s SHA-256 before loading it. Set the paths below to their locations. The ten small PNGs under `templates/` are hand-icon samples used by the screen reader.

```powershell
$modelArgs = @('--reference', '.runtime/reference', '--base', 'D:/models/v4.4.2/best_model.pt', '--checkpoint', 'D:/models/v3/selected.pt')
.\.venv\Scripts\python screen_runner/run.py @modelArgs --mode verify
```

Add `--frame saved-battle.png` to verify card recognition and one inference on a local battle screenshot.

## Play one match

Open Null’s Royale on the battle tab with the Hog deck selected. Adjust the ADB path and serial to the local emulator. Use a fresh log name for each match.

```powershell
$adb = 'C:/Program Files/Netease/MuMuPlayer/nx_main/adb.exe'
.\.venv\Scripts\python screen_runner/run.py @modelArgs --mode selected --match-kind normal --adb $adb --serial 127.0.0.1:16416 --log .runtime/match-01.jsonl
```

When `ready_for_normal_match` appears, tap the normal Battle button in the game. The runner waits up to 95 seconds to recognize a battle. It exits after the battle HUD disappears or the active time limit expires. To stop early, create `.runtime/match-01.STOP`. Add `--record-board` to save small local board images for review.

The log records each hand, estimated elixir, visible threat positions, decision time, target coordinates, and deployment confirmation. A confirmation means the game cycled the hand. Use the board images to assess whether a defender stopped the push.
