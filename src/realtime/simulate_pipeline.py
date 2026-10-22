"""
simulate_pipeline.py
---------------------
Runs the FULL pipeline end-to-end (aggregator -> predictor) using SIMULATED
eye/mouse/keyboard signals instead of real hardware. This is what we use to
test and validate the pipeline in an environment without a webcam/display
(e.g. CI, this sandbox). It exercises exactly the same FeatureAggregator and
CognitiveStatePredictor classes that main.py uses with real hardware -- only
the raw signal sources are swapped for synthetic generators.

Run:
    python3 src/realtime/simulate_pipeline.py
"""

import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.feature_extraction.feature_aggregator import FeatureAggregator
from src.model.predict import CognitiveStatePredictor


def simulate_eye_frames(true_state: str, n_frames: int, start_t: float, fps: float = 15.0, seed: int = 0):
    """Fabricate a sequence of per-frame eye measurements consistent with a
    given ground-truth cognitive state, mimicking what EyeTracker would
    produce from real webcam frames."""
    rng = np.random.default_rng(seed)
    profiles = {
        "focus":      dict(ear=0.29, blink_p=0.02, gaze_jitter=0.01, pupil=0.56),
        "fatigue":    dict(ear=0.20, blink_p=0.05, gaze_jitter=0.015, pupil=0.47),
        "overload":   dict(ear=0.26, blink_p=0.03, gaze_jitter=0.02, pupil=0.78),
        "distracted": dict(ear=0.27, blink_p=0.03, gaze_jitter=0.06, pupil=0.55),
    }
    p = profiles[true_state]
    frames = []
    gaze = np.array([0.5, 0.5])
    dt = 1.0 / fps
    for i in range(n_frames):
        t = start_t + i * dt
        ear = np.clip(rng.normal(p["ear"], 0.02), 0.1, 0.4)
        is_blink = rng.random() < p["blink_p"]
        gaze = gaze + rng.normal(0, p["gaze_jitter"], size=2)
        gaze = np.clip(gaze, 0, 1)
        is_saccade = np.linalg.norm(rng.normal(0, p["gaze_jitter"], size=2)) > 0.06
        pupil = np.clip(rng.normal(p["pupil"], 0.05), 0.2, 1.0)
        frames.append({
            "timestamp": t, "ear": ear, "is_blink": is_blink,
            "gaze_x": gaze[0], "gaze_y": gaze[1],
            "pupil_diameter": pupil, "is_saccade": is_saccade,
        })
    return frames


def simulate_input_snapshot(true_state: str, window_seconds: float, seed: int = 0):
    """Fabricate a mouse/keyboard snapshot consistent with a ground-truth
    state, mimicking InputTracker.snapshot()."""
    rng = np.random.default_rng(seed)
    profiles = {
        "focus":      dict(n_moves=40, speed=500, click_n=1, n_keys=35, backsp_p=0.05, pause_p=0.03),
        "fatigue":    dict(n_moves=20, speed=250, click_n=1, n_keys=22, backsp_p=0.10, pause_p=0.10),
        "overload":   dict(n_moves=35, speed=450, click_n=2, n_keys=30, backsp_p=0.17, pause_p=0.06),
        "distracted": dict(n_moves=15, speed=380, click_n=1, n_keys=18, backsp_p=0.11, pause_p=0.25),
    }
    p = profiles[true_state]
    now = time.time()

    positions = []
    x, y = 500, 400
    for i in range(p["n_moves"]):
        t = now - window_seconds + (i / p["n_moves"]) * window_seconds
        x += rng.normal(0, p["speed"] / 10)
        y += rng.normal(0, p["speed"] / 10)
        positions.append((t, x, y))

    clicks = list(now - window_seconds + rng.random(p["click_n"]) * window_seconds)

    keys = []
    t = now - window_seconds
    for i in range(p["n_keys"]):
        if rng.random() < p["pause_p"]:
            t += rng.uniform(3.5, 6.0)  # long pause
        else:
            t += rng.uniform(0.1, 0.5)
        if t > now:
            break
        keys.append(t)

    backspaces = [k for k in keys if rng.random() < p["backsp_p"]]

    return {
        "mouse_positions": positions,
        "mouse_clicks": clicks,
        "key_events": keys,
        "backspace_events": backspaces,
        "window_seconds": window_seconds,
    }


def main():
    predictor = CognitiveStatePredictor()
    window_seconds = 10.0
    true_states = ["focus", "fatigue", "overload", "distracted"]

    correct = 0
    total = 0
    print("=" * 70)
    print("END-TO-END PIPELINE SIMULATION")
    print("(synthetic webcam + mouse/keyboard signals -> aggregator -> model)")
    print("=" * 70)

    n_trials = 40
    for trial in range(n_trials):
        true_state = true_states[trial % len(true_states)]
        aggregator = FeatureAggregator(window_seconds=window_seconds)

        start_t = time.time() - window_seconds
        eye_frames = simulate_eye_frames(true_state, n_frames=int(window_seconds * 15),
                                          start_t=start_t, seed=1000 + trial)
        for f in eye_frames:
            aggregator.push_eye_frame(f)

        input_snapshot = simulate_input_snapshot(true_state, window_seconds, seed=2000 + trial)
        feats = aggregator.compute_features(input_snapshot)
        result = predictor.predict(feats)

        is_correct = result["state"] == true_state
        correct += int(is_correct)
        total += 1
        mark = "correct" if is_correct else "WRONG"
        print(f"\nTrial {trial+1:2d} | true={true_state:<11} pred={result['state']:<11} "
              f"conf={result['confidence']:.1%}  [{mark}]")
        probs_str = "  ".join(f"{k}:{v:.2f}" for k, v in result["all_probabilities"].items())
        print(f"           probs -> {probs_str}")

    print("\n" + "=" * 70)
    print(f"Simulation accuracy: {correct}/{total} = {correct/total:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
