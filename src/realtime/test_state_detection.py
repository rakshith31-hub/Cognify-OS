"""
test_state_detection.py
------------------------
Answers "does the `distracted` class actually work in this project?"
independent of any camera/head-pose issues -- it bypasses EyeTracker
entirely and feeds the model hand-built feature vectors that match each
class's known training profile, so you can see whether the CLASSIFIER
itself correctly separates the 4 states.

If this shows near-100% accuracy per class but your LIVE daemon still
misreads "distracted", the problem is upstream of the model -- i.e. the
features being computed from your webcam/input don't actually look like
distracted-profile features yet (see README's "About head pose" note, and
the head_pose_override in daemon.py that now covers the most common gap:
head not facing the screen).

Run: python3 src/realtime/test_state_detection.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.model.predict import CognitiveStatePredictor
from src.data_generation.generate_synthetic_data import generate, STATES


def main():
    predictor = CognitiveStatePredictor()

    print("=" * 70)
    print("1) Sanity check: does the model separate the 4 classes at all?")
    print("   (200 fresh synthetic samples per class, same distribution used")
    print("    to train it -- NOT a live/camera test)")
    print("=" * 70)
    X_seq, y, df = generate(n_samples=800, seed=12345)
    feature_cols = [c for c in df.columns if c != "cognitive_state"]

    per_class_correct = {s: 0 for s in STATES}
    per_class_total = {s: 0 for s in STATES}
    confusions = {}

    for i in range(len(df)):
        row = df.iloc[i]
        feats = {c: float(row[c]) for c in feature_cols}
        result = predictor.predict(feats)
        true_state = row["cognitive_state"]
        per_class_total[true_state] += 1
        if result["state"] == true_state:
            per_class_correct[true_state] += 1
        else:
            confusions[(true_state, result["state"])] = confusions.get((true_state, result["state"]), 0) + 1

    for s in STATES:
        acc = per_class_correct[s] / per_class_total[s] if per_class_total[s] else float("nan")
        print(f"  {s:>12s}: {per_class_correct[s]:3d}/{per_class_total[s]:3d} correct ({acc:.1%})")

    if confusions:
        print("\n  Most common confusions:")
        for (true_s, pred_s), count in sorted(confusions.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {true_s} -> predicted {pred_s}: {count}x")

    print()
    print("=" * 70)
    print("2) A single hand-built 'obviously distracted' feature vector")
    print("   (high gaze dispersion, short fixations, high mouse-idle,")
    print("    low typing speed -- the textbook distracted signature)")
    print("=" * 70)
    obviously_distracted = {
        "blink_rate_per_min": 20.0, "avg_ear": 0.27, "fixation_duration_ms": 150.0,
        "gaze_dispersion": 0.15, "saccade_rate_per_min": 40.0, "pupil_diameter_norm": 0.55,
        "mouse_speed_mean": 400.0, "mouse_speed_var": 7000.0, "click_rate_per_min": 5.0,
        "mouse_idle_ratio": 0.6, "typing_speed_kpm": 100.0, "key_interval_var": 20000.0,
        "backspace_rate": 0.1, "long_pause_count": 5.0,
    }
    result = predictor.predict(obviously_distracted)
    print(f"  Predicted: {result['state']} ({result['confidence']:.1%} confidence)")
    print(f"  All probabilities: {result['all_probabilities']}")

    print()
    print("=" * 70)
    print("If part 1 shows high per-class accuracy (especially for")
    print("'distracted'), the classifier itself works correctly -- any live")
    print("misreads are about what compute_features()/compute_sequence_features()")
    print("produce from YOUR actual webcam+input, not the model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
