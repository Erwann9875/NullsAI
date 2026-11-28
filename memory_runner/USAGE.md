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

By default the runner drags the card from its hand slot to the exact tile center, holds it there for 60 ms, then releases. Use `--placement-input tap` for separate hand and tile taps. Each `touch_sent` event records the grid cell and time spent sending the gesture. For troops and buildings, `deployment_observed` records their first visible position. Moving units and multi-unit formations can appear away from the selected center. The receipt includes the tick of the decision that submitted it.

Add `--observe-only` to read the match without loading the policy or submitting plays. Use `--adb` to select an ADB executable. Create `.runtime/match-01.STOP` to stop this run.

When starting in the lobby, pass `--start-signal PATH` and create that file when you tap the match button. At `ready`, the model is loaded and the reader has indexed candidates for this game process. It checks a moving clock, the local account and the deck before arming. With a recognized index it keeps using the fast lookup throughout the configured matchmaking wait. `armed.attach_ms` measures the successful lookup. `first_observation.from_start_ms` includes matchmaking and loading. The first verified board can produce a decision immediately. Use `--without-clock-cache` to test a fresh attachment.

## Train defensive responses

When the battle advances too far during a board read, the runner discards that frame and retries the same event cursor with a bounded backoff. The log records `waiting_for_coherent_frame` and `coherent_frame_restored`. An action always requires a validated observation.

Keep `defense-responses.npz` and `defense-context.npz` next to the existing `tile-prior.npz`. The runner loads them when it starts and reports `replay_defenses` and `context_defenses` in the ready event. The pretrained checkpoint continues to handle proactive plays. The defensive head uses the current hand, opposing card, lane and recent public plays. An unknown Tesla exposure state contributes zero Fireball value. An offensive X-Bow or Mortar is a threat when an active tower is within reach.

These commands use the cached IL_Replay shards and the upstream engine card catalog. The first pass writes compact examples for reuse on subsequent fits. Use a fresh examples file when changing the dataset or the feature encoding.

```powershell
$models = '.runtime/models'
$catalog = '.runtime/reference/engine/data/cards.json'
python memory_runner/fit_defense_responses.py $cache $catalog "$models/defense-replay.npz" --report "$models/replay-report.json"
python memory_runner/fit_response_context.py --cache $cache --catalog $catalog --examples "$models/response-examples.npz" --table "$models/defense-replay.npz" --out "$models/defense-context.npz" --report "$models/context-report.json"
```

For simulator training, `defense_trials.py` forks the same visible push into alternative legal defenses and waits 40 seconds of game time to score tower damage. Vary seeds, hand, lane, card and deployment tile. It uses one simulation worker and a 256 MB JVM heap. Set `$sim` and `$tables` to the installed CRForge checkout and its decoded tables. Set `$baseline` to the previous runner’s `memory_policy.py` and `$firstlight` to its FirstLight dependency when benchmarking. Each trials command needs a fresh results filename.

```powershell
$trial = @('--reference','.runtime/reference','--simulator',$sim,'--tables',$tables,'--responses',"$models/defense-replay.npz",'--tile-prior',"$models/tile-prior.npz")
$context = @('--context',"$models/defense-context.npz")
$compare = @('--baseline-source',$baseline,'--firstlight',$firstlight)
python memory_runner/defense_trials.py @trial --start 540000 --seeds 20 --out "$models/train-a.jsonl"
python memory_runner/defense_trials.py @trial --start 548000 --seeds 32 --out "$models/train-b.jsonl"
python memory_runner/defense_trials.py @trial @context @compare --start 650000 --seeds 12 --out "$models/validation.jsonl"
python memory_runner/defense_trials.py @trial @context @compare --start 760000 --seeds 20 --out "$models/test.jsonl"
python memory_runner/fit_defense_rollouts.py --train "$models/train-a.jsonl" "$models/train-b.jsonl" --validation "$models/validation.jsonl" --test "$models/test.jsonl" --table "$models/defense-replay.npz" --out "$models/defense-responses.npz" --report "$models/simulator-report.json"
```

The report compares the previous policy and the fitted policy on the same hands and attacks. Held-out replay agreement measures imitation. The separate simulator report measures Princess Tower HP, towers lost and elixir spent on single pushes.

### Revealed Bush Goblins

The runner distinguishes the revealed goblins from the Suspicious Bush card that spawned them. To fit the optional local response table, use JDK 21 and the installed CRForge simulator. Training compares seven defenders and holding elixir across different hands, approach distances and tower health. The comparison command runs the previous replay-based defense and the fitted defense on identical fresh scenarios. The runner loads `bush-defense.npz` beside the tile prior and reports `bush_defense` at startup.

```powershell
$bushSim = @('--classpath', "$sim/env/build/install/env/lib/*", '--tables', $tables)
python memory_runner/train_bush_defense.py @bushSim --out "$models/bush-defense.npz" --report "$models/bush-heldout.json"
python memory_runner/evaluate_bush_defense.py @bushSim --firstlight $firstlight --reference .runtime/reference --models $models --bush-table "$models/bush-defense.npz" --report "$models/bush-comparison.json"
```

To update one counter cheaply, collect extra trials for that matchup using `--threats`. Add `--refine giant skeletons PATH_TO_EXTRA_TRIALS` to the fitting command to update that response alone. Reserve a fresh seed range for testing after selecting a refinement. Use complete simulated matches against several decks to check the effect on elixir and overall results.

`evaluate_defense_matches.py` runs complete matches against scripted siege, Balloon, Giant and Hog decks. It pairs the trained policy with the frozen runner supplied through `--baseline-source` on identical shuffled hands, seeds and seats. For example, `--groups 10` produces 80 matches per policy. Use `--help` for the local asset and checkpoint paths. The simulator supplies legal actions and public unit observations throughout each match.

## Train and run the full-match policy

Keep `rollout-head.npz`, `match-strategy.json`, and the `crforge-public` class directory beside `tile-prior.npz`. Use JDK 21, JPype1 1.7.1, and the same CRForge install and tables used for training. The settings list the revealed cards that activate the Giant forecast or the lighter air response. The forecast checks a bounded shortlist using public plays, visible troop positions, tower HP and available elixir. The JVM uses one logical processor and a 256 MB heap limit.

```powershell
python -m pip install JPype1==1.7.1
Copy-Item memory_runner/match-strategy.json "$models/match-strategy.json"
New-Item -ItemType Directory -Force "$models/crforge-public" | Out-Null
javac -proc:none -cp "$sim/env/build/install/env/lib/*" -d "$models/crforge-public" memory_runner/PublicBoard.java
python memory_runner/run.py @device --simulator $sim --tables $tables --tile-prior "$models/tile-prior.npz" --log .runtime/match-02.jsonl
```

To reproduce the action-value head, set `$baseline` to the frozen runner being compared and use the same reference and model paths as the live runner. Each collection writes a new file.

```powershell
$bench = @('--reference','.runtime/reference','--firstlight',$firstlight,'--simulator',$sim,'--tables',$tables,'--base',"$models/best_model.pt",'--checkpoint',"$models/selected.pt",'--tile-prior',"$models/tile-prior.npz",'--defense-responses',"$models/defense-responses.npz",'--baseline-source',$baseline)
python memory_runner/evaluate_defense_matches.py @bench --matchups beatdown --start 2232400 --groups 7 --teacher-objective initial --horizon 170 --teacher-interval 30 --teacher-data "$models/labels-1.npy" --out "$models/teacher-1.jsonl"
python memory_runner/evaluate_defense_matches.py @bench --matchups beatdown --start 2240100 --groups 12 --teacher-objective balanced --horizon 125 --teacher-interval 20 --teacher-data "$models/labels-2.npy" --out "$models/teacher-2.jsonl"
python memory_runner/fit_rollout_head.py "$models/rollout-head.npz" "$models/labels-1.npy" "$models/labels-2.npy" --canonical --ranking --pairwise --epochs 45
python memory_runner/evaluate_defense_matches.py @bench --groups 20 --start 3196100 --out "$models/paired-test.jsonl"
```
