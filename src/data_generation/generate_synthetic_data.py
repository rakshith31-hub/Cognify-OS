"""
generate_synthetic_data.py
---------------------------
Generates synthetic training data for the cognitive-state classifier.

Unlike the original generator (which only ever produced ONE aggregated
14-feature row per window), this version generates the underlying raw
PER-SECOND signal first, as an autocorrelated (AR(1)) process around
class-conditional means/stds grounded in the original dataset's empirical
per-class statistics (and the domain rationale in docs/RESEARCH_RATIONALE.md).

From that same per-second sequence we then derive TWO artifacts so the
Random Forest (aggregated) and CNN (sequence) models are trained on
data from the same generative process and are fairly comparable:

  1. data/cognitive_state_dataset.csv
     One row per window = the 14 features aggregated the SAME way
     FeatureAggregator.compute_features() aggregates real signals:
     mean over the window for rate/level features, variance over the
     window for the two variance features (mouse_speed_var, key_interval_var).

  2. data/cognitive_state_sequences.npz
     X: (N, T, 14) float32 array -- T=10 per-second instantaneous feature
        estimates per window (what a CNN convolves over).
     y: (N,) string array of labels.
     feature_cols: the 14 feature names, in channel order.

Both keep the same 4% random label-noise policy as the original generator,
to mimic real annotation ambiguity, and it's applied identically (same
label swapped) to both artifacts for a given sample.
"""

import os
import json
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

FEATURE_COLS = [
    "blink_rate_per_min", "avg_ear", "fixation_duration_ms", "gaze_dispersion",
    "saccade_rate_per_min", "pupil_diameter_norm",
    "mouse_speed_mean", "mouse_speed_var", "click_rate_per_min", "mouse_idle_ratio",
    "typing_speed_kpm", "key_interval_var", "backspace_rate", "long_pause_count",
]
# NOTE: mouse_speed_var and key_interval_var are themselves variance-scale
# quantities (their PROFILES mean/std below are on the values as computed
# by the real FeatureAggregator, e.g. mean~4647 for focus). We model their
# per-second trajectory with the same AR(1) approach as every other feature
# and aggregate with a MEAN like everything else -- NOT by taking var() of
# the sequence, which would square an already-variance-scale quantity and
# blow up the scale by ~1000x. This keeps units consistent between the
# aggregated CSV and the sequence tensor.

STATES = ["focus", "fatigue", "overload", "distracted"]
T_STEPS = 10          # one instantaneous estimate per second of the 10s window
LABEL_NOISE_RATE = 0.04
N_SAMPLES = 12000
SEED = 42

# class-conditional (mean, std) per feature, grounded in the original
# dataset's empirical per-class statistics (see generation notebook / README
# domain rationale: low gaze dispersion + long fixations = focus, low EAR +
# high blink rate = fatigue, high pupil + high backspace = overload, high
# gaze dispersion + short fixations + idle gaps = distracted).
PROFILES = {
    "focus": {
        "blink_rate_per_min": (16.1, 6.2), "avg_ear": (0.288, 0.046),
        "fixation_duration_ms": (355.3, 92.4), "gaze_dispersion": (0.046, 0.030),
        "saccade_rate_per_min": (26.1, 9.1), "pupil_diameter_norm": (0.565, 0.123),
        "mouse_speed_mean": (521.7, 163.5), "mouse_speed_var": (4647.4, 2442.3),
        "click_rate_per_min": (7.69, 3.97), "mouse_idle_ratio": (0.208, 0.136),
        "typing_speed_kpm": (208.4, 56.4), "key_interval_var": (7770.8, 3762.4),
        "backspace_rate": (0.063, 0.041), "long_pause_count": (1.20, 1.21),
    },
    "fatigue": {
        "blink_rate_per_min": (26.6, 7.2), "avg_ear": (0.212, 0.051),
        "fixation_duration_ms": (307.9, 91.2), "gaze_dispersion": (0.060, 0.031),
        "saccade_rate_per_min": (20.5, 8.2), "pupil_diameter_norm": (0.485, 0.123),
        "mouse_speed_mean": (347.1, 134.1), "mouse_speed_var": (4316.8, 2564.9),
        "click_rate_per_min": (5.14, 2.95), "mouse_idle_ratio": (0.326, 0.147),
        "typing_speed_kpm": (155.4, 50.8), "key_interval_var": (10495.5, 4035.9),
        "backspace_rate": (0.099, 0.049), "long_pause_count": (2.76, 1.60),
    },
    "overload": {
        "blink_rate_per_min": (21.1, 6.1), "avg_ear": (0.260, 0.046),
        "fixation_duration_ms": (271.1, 82.2), "gaze_dispersion": (0.070, 0.031),
        "saccade_rate_per_min": (32.0, 9.4), "pupil_diameter_norm": (0.751, 0.140),
        "mouse_speed_mean": (455.6, 149.9), "mouse_speed_var": (7813.1, 3251.3),
        "click_rate_per_min": (8.89, 4.08), "mouse_idle_ratio": (0.227, 0.130),
        "typing_speed_kpm": (184.2, 55.8), "key_interval_var": (13018.0, 4963.9),
        "backspace_rate": (0.157, 0.062), "long_pause_count": (2.28, 1.42),
    },
    "distracted": {
        "blink_rate_per_min": (20.0, 6.1), "avg_ear": (0.271, 0.045),
        "fixation_duration_ms": (194.8, 75.7), "gaze_dispersion": (0.112, 0.045),
        "saccade_rate_per_min": (36.3, 9.2), "pupil_diameter_norm": (0.553, 0.132),
        "mouse_speed_mean": (407.6, 157.6), "mouse_speed_var": (6723.2, 3078.7),
        "click_rate_per_min": (6.16, 3.75), "mouse_idle_ratio": (0.474, 0.165),
        "typing_speed_kpm": (132.3, 51.2), "key_interval_var": (17647.1, 5617.1),
        "backspace_rate": (0.111, 0.050), "long_pause_count": (4.42, 2.22),
    },
}

# non-negative-only features (rates, counts, variances) get clipped at 0
NON_NEGATIVE = {
    "blink_rate_per_min", "fixation_duration_ms", "gaze_dispersion",
    "saccade_rate_per_min", "pupil_diameter_norm", "mouse_speed_mean",
    "mouse_speed_var", "click_rate_per_min", "mouse_idle_ratio",
    "typing_speed_kpm", "key_interval_var", "backspace_rate", "long_pause_count",
}

AR_PHI = 0.6  # AR(1) autocorrelation coefficient for within-window smoothness


def _ar1_sequence(rng, mean, std, t_steps, phi=AR_PHI):
    """Generate a T-step AR(1) process with the given marginal mean/std,
    so consecutive per-second estimates are correlated (real behavioral
    signals don't jump independently every second) rather than iid noise."""
    innovation_std = std * np.sqrt(1 - phi ** 2)
    seq = np.empty(t_steps)
    seq[0] = rng.normal(mean, std)
    for t in range(1, t_steps):
        seq[t] = mean + phi * (seq[t - 1] - mean) + rng.normal(0, innovation_std)
    return seq


def generate(n_samples: int = N_SAMPLES, t_steps: int = T_STEPS, seed: int = SEED):
    rng = np.random.default_rng(seed)

    true_labels = rng.choice(STATES, size=n_samples)
    # 4% random label noise applied to the OBSERVED label only (the signal
    # itself is still generated from the true class profile -- this mimics
    # a human rater mislabeling an otherwise-normal window)
    noisy_mask = rng.random(n_samples) < LABEL_NOISE_RATE
    observed_labels = true_labels.copy()
    for i in np.where(noisy_mask)[0]:
        others = [s for s in STATES if s != true_labels[i]]
        observed_labels[i] = rng.choice(others)

    X_seq = np.empty((n_samples, t_steps, len(FEATURE_COLS)), dtype=np.float32)
    agg_rows = np.empty((n_samples, len(FEATURE_COLS)), dtype=np.float32)

    for i in range(n_samples):
        profile = PROFILES[true_labels[i]]
        for j, feat in enumerate(FEATURE_COLS):
            mean, std = profile[feat]
            seq = _ar1_sequence(rng, mean, std, t_steps)
            if feat in NON_NEGATIVE:
                seq = np.clip(seq, 0.0, None)
            X_seq[i, :, j] = seq
            agg_rows[i, j] = np.mean(seq)

    df = pd.DataFrame(agg_rows, columns=FEATURE_COLS)
    df["cognitive_state"] = observed_labels

    return X_seq, observed_labels, df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    X_seq, y, df = generate()

    csv_path = os.path.join(DATA_DIR, "cognitive_state_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")

    npz_path = os.path.join(DATA_DIR, "cognitive_state_sequences.npz")
    np.savez_compressed(npz_path, X=X_seq, y=y, feature_cols=np.array(FEATURE_COLS))
    print(f"Wrote sequences {X_seq.shape} to {npz_path}")

    meta_path = os.path.join(DATA_DIR, "sequence_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"t_steps": T_STEPS, "feature_cols": FEATURE_COLS, "states": STATES}, f, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
