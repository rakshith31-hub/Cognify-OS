"""
compare_models.py
-------------------
Empirically compares every model variation this project has:
  - RandomForest alone (the current production default)
  - CNN alone
  - Weighted-average ensembles of RF+CNN probabilities, swept across weights
  - A stacking meta-learner (logistic regression on [rf_probs, cnn_probs])

...evaluated on TWO separate test conditions, because relying on only one
would be misleading (this project already found the CNN looks great on
clean synthetic data but degrades sharply on the live pipeline):

  A) Synthetic held-out set: the same test split each model was originally
     validated on (data/cognitive_state_dataset.csv +
     data/cognitive_state_sequences.npz, row-aligned by construction --
     see generate_synthetic_data.py). Ensemble weights/meta-learner are
     fit on HALF of this held-out set and evaluated on the other half, to
     avoid fitting and evaluating the ensemble on the same data.

  B) Live-pipeline simulation: reuses simulate_pipeline.py's synthetic
     hardware-event generator (a DIFFERENT, independently-authored
     generator from the AR1 one used for training) run through the ACTUAL
     FeatureAggregator + real predictor code path -- this is the
     harsher, more realistic condition where the earlier CNN-fragility
     finding showed up.

Prints a single comparison table across both conditions and states a
recommendation, rather than trusting either condition alone.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.model.predict import CognitiveStatePredictor, CNNCognitiveStatePredictor
from src.model.train_cnn import CognitiveStateCNN
from src.feature_extraction.feature_aggregator import FeatureAggregator
from src.realtime.simulate_pipeline import simulate_eye_frames, simulate_input_snapshot

DATA_CSV = os.path.join(_PROJECT_ROOT, "data", "cognitive_state_dataset.csv")
DATA_NPZ = os.path.join(_PROJECT_ROOT, "data", "cognitive_state_sequences.npz")
STATES = ["distracted", "fatigue", "focus", "overload"]  # LabelEncoder's alphabetical order
SEED = 42


def _probs_dict_to_array(probs: dict) -> np.ndarray:
    return np.array([probs[s] for s in STATES])


# ============================================================ condition A
def evaluate_synthetic_holdout():
    print("=" * 70)
    print("CONDITION A: synthetic held-out set (same split each model trained on)")
    print("=" * 70)

    df = pd.read_csv(DATA_CSV)
    npz = np.load(DATA_NPZ, allow_pickle=True)
    X_seq_all, y_all = npz["X"], npz["y"]
    feature_cols = list(npz["feature_cols"])
    assert len(df) == len(X_seq_all), "CSV and sequence npz must be row-aligned"

    indices = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=y_all
    )
    # further split the held-out test set in half: fit ensemble weights on
    # one half, evaluate everything (including RF/CNN alone) on the other,
    # so the ensemble numbers aren't fit and judged on the same data
    idx_ens_fit, idx_ens_eval = train_test_split(
        idx_test, test_size=0.5, random_state=SEED, stratify=y_all[idx_test]
    )

    rf = CognitiveStatePredictor()
    cnn = CNNCognitiveStatePredictor()

    def get_probs(idx_array):
        rf_probs, cnn_probs, true_labels = [], [], []
        for i in idx_array:
            feats = {c: df.iloc[i][c] for c in feature_cols}
            rf_result = rf.predict(feats)
            cnn_result = cnn.predict(X_seq_all[i])
            rf_probs.append(_probs_dict_to_array(rf_result["all_probabilities"]))
            cnn_probs.append(_probs_dict_to_array(cnn_result["all_probabilities"]))
            true_labels.append(y_all[i])
        return np.array(rf_probs), np.array(cnn_probs), np.array(true_labels)

    print(f"Computing predictions on {len(idx_ens_fit)} fit + {len(idx_ens_eval)} eval samples...")
    rf_fit, cnn_fit, y_fit = get_probs(idx_ens_fit)
    rf_eval, cnn_eval, y_eval = get_probs(idx_ens_eval)

    label_to_idx = {s: i for i, s in enumerate(STATES)}
    y_fit_idx = np.array([label_to_idx[l] for l in y_fit])
    y_eval_idx = np.array([label_to_idx[l] for l in y_eval])

    results = {}
    results["RF alone"] = accuracy_score(y_eval_idx, rf_eval.argmax(axis=1))
    results["CNN alone"] = accuracy_score(y_eval_idx, cnn_eval.argmax(axis=1))

    best_weight, best_fit_acc = None, -1
    for w in np.arange(0.0, 1.01, 0.1):
        blended_fit = w * rf_fit + (1 - w) * cnn_fit
        acc = accuracy_score(y_fit_idx, blended_fit.argmax(axis=1))
        if acc > best_fit_acc:
            best_fit_acc, best_weight = acc, w

    blended_eval = best_weight * rf_eval + (1 - best_weight) * cnn_eval
    results[f"Weighted avg (RF={best_weight:.1f}/CNN={1-best_weight:.1f}, "
            f"weight chosen on held-out fit split)"] = accuracy_score(y_eval_idx, blended_eval.argmax(axis=1))
    results["Weighted avg (fixed 50/50)"] = accuracy_score(
        y_eval_idx, (0.5 * rf_eval + 0.5 * cnn_eval).argmax(axis=1)
    )

    stack_X_fit = np.concatenate([rf_fit, cnn_fit], axis=1)
    stack_X_eval = np.concatenate([rf_eval, cnn_eval], axis=1)
    meta = LogisticRegression(max_iter=1000, multi_class="multinomial")
    meta.fit(stack_X_fit, y_fit_idx)
    results["Stacking (logistic regression meta-learner)"] = accuracy_score(
        y_eval_idx, meta.predict(stack_X_eval)
    )

    for name, acc in results.items():
        print(f"  {name:65s} {acc:.4f}")
    return results


# ============================================================ condition B
def evaluate_live_pipeline(n_trials_per_state: int = 15):
    print()
    print("=" * 70)
    print("CONDITION B: live-pipeline simulation (harsher, realistic condition)")
    print("=" * 70)

    rf = CognitiveStatePredictor()
    cnn = CNNCognitiveStatePredictor()
    window_seconds = 10.0

    rf_probs_all, cnn_probs_all, true_labels = [], [], []

    for state in ["focus", "fatigue", "overload", "distracted"]:
        for trial in range(n_trials_per_state):
            seed = hash((state, trial)) % 100000
            aggregator = FeatureAggregator(window_seconds=window_seconds)
            start_t = time.time() - window_seconds
            eye_frames = simulate_eye_frames(state, n_frames=int(window_seconds * 15),
                                              start_t=start_t, seed=seed)
            for f in eye_frames:
                aggregator.push_eye_frame(f)
            input_snapshot = simulate_input_snapshot(state, window_seconds, seed=seed + 1)

            rf_feats = aggregator.compute_features(input_snapshot)
            rf_result = rf.predict(rf_feats)

            cnn_seq = aggregator.compute_sequence_features(input_snapshot, t_steps=cnn.t_steps)
            cnn_result = cnn.predict(cnn_seq)

            rf_probs_all.append(_probs_dict_to_array(rf_result["all_probabilities"]))
            cnn_probs_all.append(_probs_dict_to_array(cnn_result["all_probabilities"]))
            true_labels.append(state)

    rf_probs_all = np.array(rf_probs_all)
    cnn_probs_all = np.array(cnn_probs_all)
    label_to_idx = {s: i for i, s in enumerate(STATES)}
    y_idx = np.array([label_to_idx[l] for l in true_labels])

    results = {}
    results["RF alone"] = accuracy_score(y_idx, rf_probs_all.argmax(axis=1))
    results["CNN alone"] = accuracy_score(y_idx, cnn_probs_all.argmax(axis=1))
    for w in [0.5, 0.7, 0.8, 0.9]:
        blended = w * rf_probs_all + (1 - w) * cnn_probs_all
        results[f"Weighted avg (RF={w:.1f}/CNN={1-w:.1f})"] = accuracy_score(y_idx, blended.argmax(axis=1))

    for name, acc in results.items():
        print(f"  {name:65s} {acc:.4f}  (n={len(y_idx)})")
    return results


if __name__ == "__main__":
    a = evaluate_synthetic_holdout()
    b = evaluate_live_pipeline()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':65s} {'Synthetic':>10s} {'Live-pipeline':>14s}")
    print(f"{'RF alone':65s} {a['RF alone']:>10.4f} {b['RF alone']:>14.4f}")
    print(f"{'CNN alone':65s} {a['CNN alone']:>10.4f} {b['CNN alone']:>14.4f}")
