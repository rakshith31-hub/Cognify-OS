"""
evaluate_ensemble.py
----------------------
Empirically answers "which model (or ensemble weight) is actually best" --
not by assumption, but by running RF alone, CNN alone, and weighted
probability-averaging ensembles across the full weight range, in TWO
regimes:

  1. Fresh in-distribution synthetic data (unseen by either model during
     training -- a new random seed). This is the "best case" test: same
     generative process both models were trained on.

  2. The live feature-extraction pipeline (FeatureAggregator fed by the
     older hand-authored raw-signal simulator in simulate_pipeline.py) --
     the harder, more realistic test, since it's not from the same
     AR(1) generative process the synthetic training data uses. This is
     where CNN's fragility to distribution shift (found earlier) should
     actually show up, if it's real.

A model/weight that only wins on (1) but loses on (2) is NOT actually
better for real use -- (2) is the closer proxy to what a live user's
webcam/mouse/keyboard signal will actually look like.
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.model.predict import CognitiveStatePredictor, CNNCognitiveStatePredictor
from src.data_generation.generate_synthetic_data import generate, STATES
from src.feature_extraction.feature_aggregator import FeatureAggregator, FEATURE_COLS
from src.realtime.simulate_pipeline import simulate_eye_frames, simulate_input_snapshot

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # fraction weight on CNN


def _ensemble_predict(rf_probs: dict, cnn_probs: dict, cnn_weight: float) -> str:
    classes = rf_probs.keys()
    combined = {c: (1 - cnn_weight) * rf_probs[c] + cnn_weight * cnn_probs[c] for c in classes}
    return max(combined, key=combined.get)


def evaluate_in_distribution(rf, cnn, n_samples=400, seed=777):
    """Fresh synthetic data, unseen by either model during training."""
    X_seq, y, df = generate(n_samples=n_samples, seed=seed)
    agg_rows = [{c: float(df.iloc[i][c]) for c in FEATURE_COLS} for i in range(n_samples)]

    rf_preds_by_weight = {w: [] for w in WEIGHTS}
    for i in range(n_samples):
        rf_out = rf.predict(agg_rows[i])
        cnn_out = cnn.predict(X_seq[i])
        for w in WEIGHTS:
            rf_preds_by_weight[w].append(_ensemble_predict(
                rf_out["all_probabilities"], cnn_out["all_probabilities"], w
            ))

    return {w: float(np.mean(np.array(rf_preds_by_weight[w]) == y)) for w in WEIGHTS}


def evaluate_live_pipeline(rf, cnn, n_trials_per_state=30, window_seconds=10.0, seed_offset=0):
    """
    Runs the SAME simulated raw window (eye frames + input events) through
    the real FeatureAggregator to get BOTH the aggregated row (what RF
    sees) and the sequence (what CNN sees) -- a fair paired comparison on
    the harder, more realistic distribution-shift test.
    """
    import time as time_mod

    rf_preds_by_weight = {w: [] for w in WEIGHTS}
    y_true = []

    trial = 0
    for true_state in STATES:
        for i in range(n_trials_per_state):
            seed = (trial + seed_offset) * 7919 + 1  # distinct, deterministic per trial
            aggregator = FeatureAggregator(window_seconds=window_seconds)
            start_t = time_mod.time() - window_seconds
            eye_frames = simulate_eye_frames(true_state, n_frames=int(window_seconds * 15),
                                              start_t=start_t, seed=seed)
            for f in eye_frames:
                aggregator.push_eye_frame(f)
            input_snapshot = simulate_input_snapshot(true_state, window_seconds, seed=seed + 1)

            agg_feats = aggregator.compute_features(input_snapshot)
            seq_feats = aggregator.compute_sequence_features(input_snapshot, t_steps=cnn.t_steps)

            rf_out = rf.predict(agg_feats)
            cnn_out = cnn.predict(seq_feats)

            for w in WEIGHTS:
                pred = _ensemble_predict(rf_out["all_probabilities"], cnn_out["all_probabilities"], w)
                rf_preds_by_weight[w].append(pred)
            y_true.append(true_state)
            trial += 1

    y_true = np.array(y_true)
    return {w: float(np.mean(np.array(rf_preds_by_weight[w]) == y_true)) for w in WEIGHTS}


def main():
    print("Loading models...")
    rf = CognitiveStatePredictor()
    cnn = CNNCognitiveStatePredictor()

    print("\n=== Regime 1: fresh in-distribution synthetic data (400 samples) ===")
    in_dist = evaluate_in_distribution(rf, cnn)
    for w in WEIGHTS:
        label = "RF only" if w == 0.0 else ("CNN only" if w == 1.0 else f"{int((1-w)*100)}/{int(w*100)} RF/CNN")
        print(f"  cnn_weight={w:.1f}  ({label:16s})  accuracy={in_dist[w]:.4f}")

    print("\n=== Regime 2: live feature-extraction pipeline (harder, more realistic; 120 trials) ===")
    live = evaluate_live_pipeline(rf, cnn)
    for w in WEIGHTS:
        label = "RF only" if w == 0.0 else ("CNN only" if w == 1.0 else f"{int((1-w)*100)}/{int(w*100)} RF/CNN")
        print(f"  cnn_weight={w:.1f}  ({label:16s})  accuracy={live[w]:.4f}")

    print("\n=== Combined verdict ===")
    print(f"{'weight':>8} {'in-dist':>10} {'live':>10} {'min(both)':>12}")
    best_w, best_min = None, -1
    for w in WEIGHTS:
        combined_min = min(in_dist[w], live[w])
        marker = ""
        if combined_min > best_min:
            best_min = combined_min
            best_w = w
        print(f"{w:>8.1f} {in_dist[w]:>10.4f} {live[w]:>10.4f} {combined_min:>12.4f}")
    label = "RF only" if best_w == 0.0 else ("CNN only" if best_w == 1.0 else f"{int((1-best_w)*100)}/{int(best_w*100)} RF/CNN")
    print(f"\nBest by worst-case (min of both regimes): cnn_weight={best_w} ({label})")
    print(f"  in-distribution accuracy: {in_dist[best_w]:.4f}")
    print(f"  live-pipeline accuracy:   {live[best_w]:.4f}")
    print(f"  (current shipped default is RF only, cnn_weight=0.0: "
          f"in-dist={in_dist[0.0]:.4f}, live={live[0.0]:.4f})")


if __name__ == "__main__":
    main()
