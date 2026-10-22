# COGNIFY OS — Real-Time Cognitive State Monitor

Infers a user's cognitive state — **focus / fatigue / overload / distracted**
(with a confidence score for each) — from real-time **webcam eye-tracking**
signals combined with **mouse and keyboard behavior**, and **runs in the
background as a daemon that acts on it**: toggling Do Not Disturb, nudging
brightness, and (opt-in) pausing self-chosen distracting sites — see
`docs/RESEARCH_RATIONALE.md` for why each action is mapped to each state.

Two model options, both trained on the same 14 features / same 4 classes:
- **RandomForest** (`models/cognitive_state_model.joblib`, default) — one
  aggregated feature vector per 10s window. `src/model/train_model.py`.
- **1D-CNN** (`models/cognitive_state_cnn.pt`) — a (10 timestep × 14
  feature) sequence per window, convolved over time so it can learn
  within-window trends the RF's mean/var aggregation discards.
  `src/model/train_cnn.py`. See **"RandomForest vs CNN"** below before
  picking one.

Two run modes:
- **`full`** — webcam + mouse/keyboard (all 14 features, requires a camera).
- **`input-only`** — mouse/keyboard only, **no camera at all**. Eye-related
  features fall back to the same neutral-default path the model already
  uses when no face is detected, so it's the exact same trained model,
  just working from 8 live features instead of 14. Use this if you don't
  want to grant camera access, don't have a webcam, or want a lighter
  footprint.

## How it works

```
Webcam ──► EyeTracker (MediaPipe FaceMesh) ──► per-frame: blink, EAR, gaze, pupil, saccade
    (optional — omitted entirely in --mode input-only)     │
Mouse/Keyboard ──► InputTracker (pynput) ──► raw events ────┤
                                                             ▼
                                    FeatureAggregator (10s sliding window)
                                                             │
                                                             ▼
                              14-dim feature vector ──► RandomForestClassifier
                                                             │
                                                             ▼
                          {state, confidence, probability distribution}
                                                             │
                                                             ▼
                    ActionEngine (hysteresis + cooldown) ──► notifications.py / ui_controller.py
                                                             │
                                       Do Not Disturb, brightness, optional hosts-file blocking,
                                       all logged to logs/state_history.csv, all auto-reverted on stop
```

### The 14 features
| Modality | Features |
|---|---|
| Eye (webcam) | blink_rate_per_min, avg_ear, fixation_duration_ms, gaze_dispersion, saccade_rate_per_min, pupil_diameter_norm |
| Mouse | mouse_speed_mean, mouse_speed_var, click_rate_per_min, mouse_idle_ratio |
| Keyboard | typing_speed_kpm, key_interval_var, backspace_rate, long_pause_count |

These are computed over a rolling **10-second window**, matching how the
model is trained (windowed, not per-frame).

### Why the 4 states separate this way (domain rationale)
- **Focus**: steady gaze (low dispersion), long fixations, consistent typing cadence, low backspace rate.
- **Fatigue**: lower eye-aspect-ratio (drooping eyelids), higher blink rate, slower typing/mouse movement, more/longer pauses.
- **Overload**: larger pupil diameter (cognitive-load proxy), high backspace/error rate, jerky/high-variance mouse movement.
- **Distracted**: high gaze dispersion, short fixations, long idle gaps interleaved with activity bursts.

## About the training data — please read

**There is no public dataset pairing raw webcam + mouse + keyboard telemetry
with ground-truth focus/fatigue/overload/distracted labels** (existing
eye-tracking cognitive-load datasets use different modalities like EEG/fNIRS,
different label sets, or aren't freely redistributable). So `data/cognitive_state_dataset.csv`
is **synthetically generated** (`src/data_generation/generate_synthetic_data.py`),
with class-conditional distributions grounded in published eye-tracking/HCI
findings (see rationale above), deliberately overlapping/noisy (not
trivially separable), plus 4% random label noise to mimic real annotation
ambiguity.

**This means the trained model demonstrates the full pipeline and a working
classification approach, but it is not validated on real human behavioral
data.** This gap is no longer purely theoretical — see **"Closing the
synthetic-data gap: real labels + retraining"** below for the actual
mechanism that collects real labels and retrains on them, which implements
items 1-2 below. Before trusting predictions on your own behavior:
1. ~~Collect real labeled data~~ — **done automatically**: `cognitive-monitor label <state>` (see below).
2. ~~Retrain on that data using the exact same pipeline~~ — **done automatically**: `cognitive-monitor retrain` (see below).
3. Re-validate per-user, since baseline blink rate/typing speed vary a lot person-to-person — see "Per-user calibration for all 14 features" below, which handles this separately from retraining.

## Closing the synthetic-data gap: real labels + retraining

The gap above is real and this doesn't pretend otherwise, but it's no longer
purely theoretical:

- **`cognitive-monitor label <state>`** attaches a real ground-truth label to
  whatever window the daemon most recently computed (kept fresh in
  `logs/last_features.json` on every tick, valid for `label_max_age_seconds` —
  default 120s). Every real label is logged to `logs/real_labels.csv` with the
  exact same 14-feature schema as the synthetic training CSV.

  Rather than scattered one-off reminders (which need standing vigilance to
  self-report at random times over days — genuinely low-feasibility, and
  worth being honest about), labeling runs as a **bounded ~5-minute session**:
  every `label_session_interval_minutes` (default 120 min), a burst of quick
  prompts fires roughly every `label_session_prompt_interval_seconds`
  (default 50s) for `label_session_duration_minutes` (default 5), then goes
  quiet until the next scheduled session. On Windows with BurntToast
  installed, these are real **tap-to-label buttons**
  (Focus/Fatigue/Overload/Distracted — no typing needed, reusing the same
  `cogmonitor://` protocol-handler infrastructure as the action-feedback
  buttons); otherwise a plain text reminder with the CLI command. All of it
  is `label_prompt_enabled: true` by default and completely ignorable — a
  session that gets no taps just ends and tries again next time.
- **`cognitive-monitor retrain`** (`src/model/retrain_from_real_data.py`)
  blends your real labels with the synthetic data — but **upweights real rows**
  proportional to how outnumbered they are (capped at 5x), since a few dozen
  real labels would otherwise be statistically drowned out by thousands of
  synthetic ones and change nothing. It evaluates on a **real-only held-out
  split**, tracked separately from the synthetic-data numbers above — that's
  the number that actually answers "does this work on you", which the
  synthetic numbers structurally can't. Needs at least 30 total real labels
  across 2+ classes (8+ each) before it'll do anything; tells you exactly how
  many more you need otherwise.
- **It won't silently make things worse**: each retrain is compared against
  the previous one's real-holdout accuracy (`models/retrain_history.json`),
  and a regression of more than 10 points refuses to deploy without `--force`.
  The previous model is backed up (`models/backups/<timestamp>/`) before
  every deploy regardless.
- **`auto_retrain_on_startup`** (config, off by default) checks once at daemon
  startup — never mid-session — whether enough new real labels have
  accumulated since the last retrain and runs it in the background
  automatically if so.

**What this doesn't do**: retrain the CNN (`train_cnn.py`) — that needs
10-timestep sequences per label, and `cognitive-monitor label` only captures
one aggregated window per label (matching what `compute_features()` gives the
RF path). Extending real-label collection to sequences would be a separate,
larger piece of work.

## Per-user calibration for all 14 features

Behavioral baselines vary a lot person-to-person — someone's naturally-low
resting blink rate shouldn't read as fatigue just because it's below the
population average. `personal_calibration_enabled` (default on) learns YOUR
OWN mean/std per feature over time (`logs/personal_feature_baseline.json`,
persists and keeps accumulating across restarts — `personal_calibration_min_windows`,
default 30, before a given feature starts adjusting) and maps incoming
readings to the **equivalent position in the population scale** the trained
model actually expects:

```
adjusted = population_mean + (raw - your_mean) / your_std * population_std
```

This is deliberately NOT naive z-scoring against your own stats alone — the
model was trained on population-scale data, so feeding it raw personal
z-scores would put it in a completely different space than it was trained on
and predictions would become meaningless. See
`src/feature_extraction/personal_calibrator.py`'s docstring for the full
reasoning, including the zero-variance fallback (a feature that's been
constant for you, e.g. truly 0 backspaces every window, falls back to a pure
mean-shift instead of dividing by ~0). This is the same idea already applied
to head pose above, extended to the other 13 features that actually feed the
classifier — head pose was rule-based and calibrated separately since it
never goes through the trained model at all.



- **5-fold CV accuracy:** 89.2% ± 0.6%
- **Held-out test accuracy:** 88.5%
- **Mean confidence on correct predictions:** 77.9%
- **Mean confidence on incorrect predictions:** 60.9% (model is meaningfully less confident when wrong — confidence score is informative, not just decorative)

```
              precision    recall  f1-score   support
  distracted      0.905     0.905     0.905       601
     fatigue      0.877     0.899     0.888       597
       focus      0.882     0.898     0.890       599
    overload      0.873     0.836     0.854       603
```

Full report: `outputs/classification_report.txt`
Confusion matrix: `outputs/confusion_matrix.png`
Feature importance: `outputs/feature_importance.png` (top drivers: pupil diameter & backspace rate → overload, gaze dispersion & mouse idle ratio → distraction, EAR & blink rate → fatigue)

**End-to-end pipeline simulation** (`src/realtime/simulate_pipeline.py`, exercises the real `FeatureAggregator` + `CognitiveStatePredictor` classes with synthetic sensor streams instead of live hardware — this is the closest thing to an integration test without a physical webcam): **35/40 = 87.5%** accuracy, consistent with the offline test accuracy.

## RandomForest vs CNN

`data/cognitive_state_dataset.csv` and `data/cognitive_state_sequences.npz` are
now generated together, from the same underlying per-second AR(1) process
(`src/data_generation/generate_synthetic_data.py`) so the two models are
trained on the same generative process and are fairly comparable — the CSV
is the per-window mean of that sequence (what RF sees), the .npz is the raw
sequence itself (what the CNN sees).

|  | RF (`train_model.py`) | CNN (`train_cnn.py`) |
|---|---|---|
| Input | 1 aggregated row / window | 10×14 raw sequence / window |
| 5-fold CV accuracy | 95.4% ± 0.4% | 95.7% ± 0.3% |
| Held-out test accuracy | 95.5% | 95.8% |
| Mean confidence, correct | 0.901 | 0.944 |
| Mean confidence, incorrect | 0.856 | 0.906 |

**Read this gap honestly, not as "CNN wins":**
- The CNN's edge here (95.8% vs 95.5%) is small. Regenerating the data as an
  autocorrelated sequence (needed for the CNN to have real temporal
  structure to convolve over) also *smoothed* the aggregated features RF
  sees, versus the original single-draw-per-row generator — both models'
  accuracy jumped from the original 88.5% mostly because of that smoothing,
  not because either model got fundamentally better at the task.
- The CNN is **more sensitive to input distribution shift** than RF. Feeding
  it live sequences from `src/realtime/simulate_pipeline_cnn.py` (which
  reuses the older hand-authored raw-signal simulator, not the AR(1)
  generator) drops accuracy well below the RF equivalent
  (`simulate_pipeline.py`), because that simulator's saccade-detection
  threshold under-fires and produces a `fixation_duration_ms` far outside
  either model's training range — RF tolerates that kind of single-feature
  outlier better than the CNN's smooth-sequence assumption does. This is a
  real, useful finding: **before trusting the CNN in production, validate
  it on the actual live feature pipeline (or real data), not just the
  offline test split** — the same warning the original README already
  gives about the RF and real-world data applies more strongly here.
- Given the small accuracy gap and the CNN's extra sensitivity, **RF alone
  remained the safer choice between the two individually** — but see
  "RandomForest + CNN ensemble" below, which changes this conclusion.

To use the CNN instead of RF standalone:
```python
from src.model.predict import CNNCognitiveStatePredictor
from src.feature_extraction.feature_aggregator import FeatureAggregator

predictor = CNNCognitiveStatePredictor()
aggregator = FeatureAggregator(window_seconds=10)
# ... push_eye_frame(...) as normal ...
sequence = aggregator.compute_sequence_features(input_snapshot, t_steps=predictor.t_steps)
result = predictor.predict(sequence)  # same {state, confidence, all_probabilities} shape
```

## RandomForest + CNN ensemble

`src/model/evaluate_ensemble.py` empirically tests RF alone, CNN alone, and
weighted probability-averaging ensembles across the full weight range, in two
regimes — fresh in-distribution synthetic data, and the harder live
feature-extraction pipeline (the more realistic proxy, since it's not from
the same generative process the synthetic training data uses).

**Result, run twice independently at different sample sizes (120 and 240
trials) with no seed overlap between runs — same pattern both times:**

| | RF alone | CNN alone | 80/20 RF/CNN ensemble |
|---|---|---|---|
| In-distribution accuracy | 97.0% | 97.0% | 97.0% |
| Live-pipeline accuracy (run 1, n=120) | 90.0% | 74.2% | **94.2%** |
| Live-pipeline accuracy (run 2, n=240) | 82.1% | 74.6% | **92.9%** |

This was not the expected outcome — CNN alone is consistently the *worst*
performer on the harder test (matching the fragility already documented
above), yet blending in a small amount of it (10-30%, optimal around 20%)
consistently **beats both individual models**, not just splits the
difference. The likely explanation: CNN captures some complementary temporal
signal that corrects RF's specific mistakes, without being weighted heavily
enough for CNN's own weaknesses to dominate. This is genuinely useful signal,
not noise — the direction and rough optimal weight replicated across two
independent large samples — but re-run `evaluate_ensemble.py` if the
underlying models are ever retrained, since the optimal weight isn't
guaranteed to stay the same.

**To use it**, set in config:
```yaml
general:
  model_type: "ensemble"       # "rf" (default) | "cnn" | "ensemble"
  ensemble_cnn_weight: 0.2      # 80/20 RF/CNN — the empirically-best weight above
```
Needs `torch` installed (`pip install torch`) and runs both models every
tick — for a 10-second prediction interval this isn't a meaningful latency
concern, but it's a real trade-off against the RF-only default's lighter
footprint. `model_type: "rf"` stays the default rather than switching to
ensemble automatically, since this is a meaningful behavior + dependency
change worth opting into deliberately rather than inheriting silently.

```
`cli.py` / `daemon.py` still default to the RF predictor; switching the
daemon's default model is a small change in `src/realtime/daemon.py` but
isn't done automatically, given the caveats above.

## Files

```
cognitive_state_project/
├── cli.py                                    # <-- the command you actually run (cognitive-monitor)
├── pyproject.toml                            # `pip install -e .` gives you the `cognitive-monitor` command
├── config/
│   └── default_config.yaml                  # what happens at each state, and the thresholds that gate it
├── docs/
│   └── RESEARCH_RATIONALE.md                # citations for why each action maps to each state
├── scripts/
│   ├── install_linux_systemd.sh             # autostart at login (systemd --user)
│   ├── install_macos_launchd.sh             # autostart at login (launchd agent)
│   └── install_windows_taskscheduler.ps1    # autostart at login (Task Scheduler)
├── data/
│   └── cognitive_state_dataset.csv          # 12,000-row synthetic training set
├── models/
│   ├── cognitive_state_model.joblib         # trained RandomForestClassifier
│   ├── scaler.joblib                        # StandardScaler
│   ├── label_encoder.joblib
│   └── feature_columns.json
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── logs/                                     # created at runtime: daemon.log, daemon.pid, state_history.csv
├── src/
│   ├── data_generation/generate_synthetic_data.py
│   ├── model/
│   │   ├── train_model.py                   # train + evaluate + save
│   │   └── predict.py                       # CognitiveStatePredictor class
│   ├── feature_extraction/
│   │   ├── eye_tracker.py                   # webcam -> MediaPipe FaceMesh signals
│   │   ├── input_tracker.py                 # pynput mouse/keyboard listener
│   │   └── feature_aggregator.py            # raw signals -> 14-dim feature vector
│   ├── actions/
│   │   ├── action_engine.py                 # state -> action, with hysteresis + cooldowns
│   │   ├── notifications.py                 # cross-platform toast + Do Not Disturb toggle
│   │   ├── ui_controller.py                 # brightness nudges + opt-in hosts-file domain blocking
│   │   └── state_logger.py                  # appends every prediction to logs/state_history.csv
│   └── realtime/
│       ├── daemon.py                        # the actual background service (used by cli.py)
│       └── simulate_pipeline.py             # end-to-end test with synthetic signals, no hardware needed
├── requirements.txt
└── README.md
```

## Config validation + guided setup

`config/default_config.yaml` stays untouched as a documented reference — every
setting in it has an inline "Rationale:" comment. Don't hand-edit it directly;
either run the wizard below, or copy it and edit the copy.

**Guided setup (recommended for first-time setup):**
```bash
cognitive-monitor setup
```
Walks through mode (full/input-only), site blocking (with domain list), the
flash-dim deterrent, dashboard auto-open, and OS-specific notes (BurntToast /
Color Filters on Windows, Focus Assist limitations on macOS). Writes to
`config/user_config.yaml`, preserving all of the template's inline comments
(via `ruamel.yaml`, not a plain re-serialize), and validates the result before
declaring success — the wizard can't hand you back a broken config.

**Just validate an existing config** (catches typo'd action types, both
blocking mechanisms enabled at once, out-of-range thresholds, unrecognized
top-level keys from bad indentation — all things that parse fine as YAML but
run silently wrong for hours):
```bash
cognitive-monitor validate-config --config config/user_config.yaml
```
`cognitive-monitor start` also runs this automatically and refuses to launch
on a hard error (bad ranges, missing sections) while warning-but-proceeding on
softer issues (unknown action type, no actions configured for a state).

## How to run

### 1. Install
```bash
pip install -e .                 # installs the `cognitive-monitor` command
pip install -r requirements.txt  # camera + input-capture dependencies
```
(On Linux you may also need `sudo apt-get install python3-tk libnotify-bin brightnessctl` for OpenCV, desktop notifications, and brightness control respectively — all optional, the daemon degrades gracefully if any are missing.)

### 2. Test the pipeline without a webcam (recommended first step)
```bash
python3 src/realtime/simulate_pipeline.py
```
This runs the real aggregator + model on synthetic sensor streams and prints a full accuracy report — good for confirming your environment is set up correctly, no camera/mouse/keyboard hardware needed.

### 3. Run it
```bash
cognitive-monitor start                        # full mode: webcam + mouse/keyboard, runs in the background
cognitive-monitor start --mode input-only       # no camera at all — mouse/keyboard only
cognitive-monitor start --foreground --verbose  # stay attached, for debugging
cognitive-monitor status                        # is it running?
cognitive-monitor logs -f                       # tail the live log
cognitive-monitor stop                          # stop it — reverts DND/brightness/any domain block first
```
`start` detaches immediately and returns your shell; state + confidence get logged to
`logs/daemon.log` and every prediction is appended to `logs/state_history.csv` for you to review later.
If camera init fails for any reason in `full` mode (no webcam, permissions denied, mediapipe not installed),
the daemon automatically falls back to `input-only` rather than crashing.

**Run it every time you log in**, instead of manually each time:
```bash
./scripts/install_linux_systemd.sh --mode full        # Linux
./scripts/install_macos_launchd.sh --mode full         # macOS
.\scripts\install_windows_taskscheduler.ps1 -Mode full # Windows (elevated PowerShell)
```

**Configure what it actually does** by editing `config/default_config.yaml` — every default
action (Do Not Disturb on focus, break suggestion + dimming on fatigue, decluttering nudge on
overload, notification pause + optional site-blocking on distraction) is explained and cited in
`docs/RESEARCH_RATIONALE.md`. Domain blocking is off by default and only ever touches domains
you explicitly list yourself.

**Permissions needed for mouse/keyboard capture (`pynput`):**
- **macOS:** System Settings → Privacy & Security → grant "Input Monitoring" and "Accessibility" to your terminal/Python.
- **Linux:** works out of the box on X11; Wayland compositors may restrict global input listeners.
- **Windows:** works out of the box.

**Permissions needed for camera capture (`full` mode only):**
- **macOS:** System Settings → Privacy & Security → Camera → grant access to your terminal/Python.
- **Linux/Windows:** usually prompted automatically on first access.

**Permissions needed for system actions:**
- Do Not Disturb / Focus toggling and brightness control use standard user-level OS tools where
  available (see `docs/RESEARCH_RATIONALE.md` and inline comments in `src/actions/` for the exact
  mechanism per OS — some, like macOS Focus, require a one-time manual Shortcut setup since Apple
  doesn't expose a scriptable API).
- The optional hosts-file domain blocking needs to write `/etc/hosts` (or the Windows equivalent),
  which needs admin/root — it's skipped with a clear log message if not available, everything else
  still works.

### 4. Regenerate data / retrain the model (optional — trained artifacts are already included)
```bash
python3 src/model/train_model.py
```

### 5. Use the predictor standalone (e.g. to integrate into your own app)
```python
from src.model.predict import CognitiveStatePredictor

predictor = CognitiveStatePredictor()
result = predictor.predict({
    "blink_rate_per_min": 16, "avg_ear": 0.29, "fixation_duration_ms": 360,
    "gaze_dispersion": 0.045, "saccade_rate_per_min": 26, "pupil_diameter_norm": 0.56,
    "mouse_speed_mean": 520, "mouse_speed_var": 4500, "click_rate_per_min": 7.5,
    "mouse_idle_ratio": 0.20, "typing_speed_kpm": 210, "key_interval_var": 7500,
    "backspace_rate": 0.06, "long_pause_count": 1.0,
})
print(result)
# {'state': 'focus', 'confidence': 0.957, 'all_probabilities': {...}}
```

## Idle/away auto-pause

Genuine AFK time (stepped away, lunch, a walk) used to still get fed through the
classifier — no input and no face just means every one of the 14 features hits
its neutral fallback, producing noisy low-confidence predictions across random
classes until `low_confidence_revert_after` eventually reverted the stale state.
`idle_pause_enabled` (default on) checks BEFORE calling the model at all: once
there's been no mouse/keyboard input **and** no face detected (in full mode —
input-only mode judges on input alone, since there's no camera signal) for
`idle_pause_after_minutes` (default 5), predictions pause entirely — no model
calls, no actions firing — and any currently-active state (DND, brightness dim,
etc.) is immediately reverted, the same as a real state change would trigger.
The moment either signal returns, predictions resume automatically.

Idle windows are logged as their own `idle` entry in `state_history.csv`, kept
**separate** from the 4 real states in the dashboard — idle time is excluded
from the focus/fatigue/overload/distracted percentages (so "60% focused" means
60% of active time, not diluted by however long you were away) and shown as its
own count instead.

## Head-pose override for "distracted" (bending your head down / looking away)

The 14-feature ML classifier has **no head-pose concept at all** — `gaze_dispersion`
only measures eyeball movement *within the socket*, so bending your head down to a
phone, or turning to look away, reads as a steady low-dispersion signal that looks
more like `focus` than `distracted` to the model. A rule-based override in
`daemon.py` runs ahead of the classifier and replaces its output with `distracted`
directly when your head hasn't been reasonably facing the screen for a
configurable fraction of the window (`head_pose_off_screen_ratio_threshold`,
default 50%), based on yaw/pitch estimated per-frame via `cv2.solvePnP` in
`eye_tracker.py`.

**This does NOT use a fixed absolute angle.** `estimate_head_pose()` uses a
generic, non-per-person 3D face model, so "facing the screen normally" reads at a
different raw angle for everyone depending on face shape and camera placement —
thresholding against 0° would make slight movements trip the override for anyone
whose neutral pose already sits close to the threshold. Instead, each session
collects a short calibration sample (`head_pose_calibration_frames`, default 30
frames, a couple seconds) to establish *your* neutral yaw/pitch, and the override
only fires on deviation from that baseline. A rolling-median smoothing window
(`head_pose_smoothing_window`, default 5 frames) also dampens single-frame
landmark-detection jitter that isn't real movement. The override doesn't activate
until calibration finishes (logged once as `Head-pose calibrated: neutral yaw=...
pitch=...`).

To verify this is working (or to tune it): run `python3 src/feature_extraction/eye_tracker.py`
standalone and watch the printed yaw/pitch while moving naturally vs. deliberately
turning away — `head_yaw_threshold_deg`/`head_pitch_threshold_deg` in
`config/default_config.yaml` control how much deviation counts as "off-screen".

**Separately**, to check the ML classifier itself can recognize a "distracted"-shaped
feature vector (independent of head pose, webcam, or lighting): run
`python3 src/realtime/simulate_pipeline.py` and check the `true=distracted` trials
in the output — this feeds synthetic feature vectors matching the trained profile
straight to the model, isolating "does the model work" from "does my webcam setup
produce the right signal".

## This app's own notifications bypass Do Not Disturb

`dnd_on`/`dnd_off` toggle DND for **other** apps (best-effort, OS-limited — see
Known Limitations). This app's **own** notifications (`notify`, `break_popup`,
the feedback prompt) are handled separately and are designed to still appear
even while DND is active:

- **Windows**: every toast this app sends uses Microsoft's documented
  `scenario="alarm"` toast attribute — the same mechanism a phone alarm-clock
  app relies on to still ring during Do Not Disturb. This is a real, documented
  Focus Assist exemption, not a workaround, and no longer depends on BurntToast
  being installed (a native `Windows.UI.Notifications` call handles it
  directly; BurntToast is now only a fallback if that somehow fails, and the
  interactive feedback-buttons toast tries the same `-Scenario Alarm` flag
  with a graceful fallback for older BurntToast versions that predate it).
- **Linux**: `dnd_on` is a no-op at the OS level by design (see Known
  Limitations) — GNOME's only scriptable DND mechanism (`gsettings
  show-banners`) is global with no per-app exception, so using it would have
  suppressed this app's own alerts right along with everyone else's. Silencing
  ourselves to silence others wasn't the right trade, so this now leaves
  GNOME's banner setting alone entirely.
- **macOS**: if you've set up the optional Focus Shortcut, macOS's own Focus
  filtering can still hide this app's notifications unless you explicitly
  allow it — go to **System Settings > Focus > [your Focus mode] > Apps** and
  add the app/terminal this daemon runs from. There's no public API to do
  this from a script; it's a one-time manual step, same category as the
  BurntToast install or the Color Filters shortcut setting.

## Site blocking, and what happens when you visit a blocked site anyway

Two separate blocking mechanisms exist, and they're not meant to be combined:

- **`distracted.actions[].block_domains`** (per-state, default off) — only blocks
  once the `distracted` state is confidently detected for several consecutive
  windows, and auto-unblocks after `duration_minutes`.
- **`site_blocking`** (top-level, default off) — blocks the listed domains for the
  **entire session**, starting the moment the daemon launches, **regardless of
  cognitive state**. This is almost always what you actually want for "don't let
  me open YouTube while I'm working" — set `site_blocking.enabled: true` and list
  domains under `site_blocking.domains`, and leave the per-state `block_domains`
  action `enabled: false`.

Blocking itself works by routing the domain to `127.0.0.1` in the OS hosts file
(`ui_controller.block_domains()`), served by `blocked_site_server.py` (a real "you
blocked this yourself" page over HTTP; HTTPS can't serve a real page without
forging a certificate, so it logs the attempt via the TLS SNI field and closes the
connection — the browser shows its own security warning, which is expected). If
blocking doesn't seem to work, check in this order:
1. Is `enabled: true`, not just `false` with a "flip to true" comment?
2. Is the daemon running as **Administrator**? Writing the hosts file needs
   elevated privileges on Windows; failures are logged, not loud.
3. Is your browser using **Secure DNS / DNS-over-HTTPS**? Chrome/Edge/Firefox
   default to this now, which resolves domains directly and skips the OS hosts
   file entirely — this is the most common reason hosts-file blocking silently
   "doesn't work" on a modern browser even when the hosts file is correct. Turn it
   off in your browser's privacy/security settings, or test with `curl`/`ping`
   first to isolate OS-level blocking from browser-level DNS behavior.

**On an actual visit attempt**, `site_blocking.dim_on_attempt` (default on) cuts
display brightness to `dim_percent` (default 5, i.e. a ~95% reduction) for
`dim_duration_seconds` (default 4s) as an immediate deterrent — not just the
passive block page, an active "you shouldn't be here right now" flash — then
restores whatever brightness was on screen right before (which may itself be a
fatigue/overload dim already in progress, not necessarily your original baseline).

## End-of-session dashboard

When the daemon stops (`cognitive-monitor stop`, Ctrl+C, or SIGTERM), a browser
tab opens automatically to a summary of just that session (`open_dashboard_on_exit:
true` in config, on by default) showing:
- State breakdown for the session (not "today", which could span several earlier runs)
- **Every notification/alarm the app itself fired**, regardless of whether Windows
  actually displayed it — there's no public API to query what Focus Assist
  suppressed while DND was on, so this is a practical proxy for "what you might
  have missed", not a true system-wide notification history.
- Blocked-site attempts, if `site_blocking` or the per-state action is enabled.
- A short list of **rule-based "changes to consider"** derived directly from that
  session's own numbers (e.g. "distracted was 40% of windows — consider enabling
  site_blocking", or "you've rejected `alarm` for `distracted` 4/5 times — consider
  removing it from that state's actions") — each recommendation names the specific
  number behind it, so it's checkable against the data on the same page, not a
  black box.

**Charts** (Chart.js, vendored locally at `src/dashboard/static/chart.umd.min.js`
— not loaded from a CDN, consistent with the no-network-calls privacy stance
below): a state-distribution doughnut, a "State Over Time" stacked bar chart
(time-bucketed, adapts bucket width to the actual span so it's useful for both
a short session and an "all time" view), an app-usage bar chart, and a
bandit-acceptance-rate bar chart color-coded by tier (green/amber/red, matching
the existing table's pills).

Run `cognitive-monitor dashboard` any time to see the same view for "today" or "all time" instead of a single session.

## Known limitations
- Synthetic training data (see above) — real-world accuracy will differ until retrained on real labeled sessions.
- No face / low light / glasses glare can degrade `EyeTracker` signal quality; the aggregator falls back to neutral defaults when no face is detected, which will bias predictions toward the model's "neutral" region rather than reflect a true state. `low_confidence_revert_after` (default 3 consecutive low-confidence windows, ~30s) reverts any stale committed state rather than leaving old actions (DND, dimming, etc.) active indefinitely once this happens — but it reverts to *no state*, it does not relabel that period as `distracted`.
- Behavioral baselines (typing speed, blink rate) vary a lot across individuals — a production system should calibrate per-user (e.g. z-score features against that user's own 1st-week baseline) rather than use global thresholds. The head-pose override (see above) already does this per-session; the 14 trained ML features do not.
- 4-class softmax confidence is a reasonable relative signal (validated above: correct predictions get higher confidence than incorrect ones) but is not a calibrated probability — if you need calibrated probabilities, wrap the classifier with `sklearn.calibration.CalibratedClassifierCV`.
- `--mode input-only` runs on 8 of the 14 features the model was trained on (mouse+keyboard only); the other 6 use neutral fallback values (the same code path as "no face detected" in full mode). Expect lower real-world accuracy than the reported 88.5% until/unless you retrain a model specifically on an 8-feature input-only subset.
- OS-level Do Not Disturb toggling is inherently platform-limited (see `src/actions/notifications.py` comments): macOS Focus and Windows Focus Assist have no public scripting APIs as of this writing, so `dnd_on` doesn't suppress OTHER apps' notifications on either platform by default — macOS needs a one-time manual Shortcut setup, Windows has no scriptable path at all. Linux/GNOME's `gsettings show-banners` toggle DOES work, but is deliberately NOT used by default anymore: it's a global switch with no per-app exception, so it would suppress this app's own notifications along with everyone else's — see the DND-bypass section below for why that matters. This app's OWN notifications are handled separately (see below) and are designed to still appear on Windows regardless of Focus Assist state.
- Windows' Color Filters accessibility feature (used for the grayscale "distracted" intervention) has no supported programmatic toggle — writing its registry key alone is silently ignored by the OS shell. `set_color_filter()` works around this by simulating the feature's own Ctrl+Win+C shortcut, which requires **"Allow the shortcut key to toggle color filters"** to be turned on once in Settings > Accessibility > Color filters — a manual step no script can enable on your behalf. If grayscale still doesn't visibly apply, check that setting first.
- The Windows balloon-tip notification fallback (used when BurntToast isn't installed) needs the underlying PowerShell process to stay alive briefly after calling `ShowBalloonTip()` for Windows to actually render it before the icon is disposed — fixed here, but a reminder that this fallback is inherently less reliable than `Install-Module -Name BurntToast -Scope CurrentUser` (no admin needed).

## Privacy & safety notes
- Everything runs **locally** — no telemetry, no network calls, no data leaves the machine. `logs/state_history.csv` is a plain local file you can inspect, edit, or delete anytime.
- The daemon reverts every system change it makes (Do Not Disturb, brightness, hosts-file edits) automatically on clean shutdown (`cognitive-monitor stop`, Ctrl+C, or SIGTERM) — see `ActionEngine.shutdown()`.
- Domain blocking is **off by default** and, even when enabled, only ever touches domains you explicitly list in `config/default_config.yaml` — this tool ships with no baked-in blocklist. This includes `site_blocking`, the session-wide mechanism — it still only ever touches domains you named yourself.
- The end-of-session dashboard opens a local Flask server on `127.0.0.1` only (not exposed to your network) and reads only this tool's own local log files.
- This monitors *your own* device's *your own* webcam/mouse/keyboard for *your own* productivity use — it's not designed or intended for monitoring anyone else's device.

