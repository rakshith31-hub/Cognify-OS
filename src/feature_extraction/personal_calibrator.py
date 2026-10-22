"""
personal_calibrator.py
------------------------
Per-user calibration for all 14 ML-trained features (blink rate, typing
speed, etc.) -- the same idea already applied to head pose in
feature_aggregator.py (calibrate a personal baseline, judge deviation from
THAT instead of a fixed global value), extended to the features that
actually feed the classifier.

WHY THIS ISN'T SIMPLE Z-SCORING
--------------------------------
The trained RF/CNN models expect features on the POPULATION's scale (the
synthetic training data's distribution) -- predict.py's
CognitiveStatePredictor loads a scaler.joblib fit on that population data.
If we z-scored incoming features against the user's OWN mean/std and fed
THAT to the model, we'd be handing it data in a completely different space
than it was trained on (a z-score isn't the same number in two different
distributions) and predictions would become meaningless.

Instead, this maps a user's own reading onto the EQUIVALENT position in
the population's distribution:

    adjusted = population_mean + (raw - user_mean) * (population_std / user_std)

This is standard distribution-matching / covariate-shift correction: it
expresses "how many of MY OWN std-devs this reading is from MY OWN mean"
and re-renders that same relative position using the population's mean and
spread -- so the model still receives values on the scale it was trained
on, but a user whose personal baseline is naturally offset (e.g. a
naturally lower resting blink rate) doesn't get systematically
misclassified just because their "normal" doesn't match the population's
"normal".

When calibration hasn't accumulated enough samples yet (or a feature's
user_std is ~0, e.g. zero backspaces the whole calibration period), this
transform falls back to identity / population values respectively --
never divides by ~0, never blows up a single quiet feature into an
extreme outlier.

PERSISTS ACROSS SESSIONS: the baseline is saved to logs/ and accumulates
(via Welford's online algorithm, merged across restarts) rather than
re-calibrating from scratch each session -- more sessions of data means a
more representative personal baseline, not a repeated few-minutes guess.
"""

import json
import math
import os

FEATURE_COLS = [
    "blink_rate_per_min", "avg_ear", "fixation_duration_ms", "gaze_dispersion",
    "saccade_rate_per_min", "pupil_diameter_norm",
    "mouse_speed_mean", "mouse_speed_var", "click_rate_per_min", "mouse_idle_ratio",
    "typing_speed_kpm", "key_interval_var", "backspace_rate", "long_pause_count",
]

# Population reference (mean, std across ALL 4 classes combined, from the
# training data) -- this is the scale the trained model actually expects.
# Loaded once from data/cognitive_state_dataset.csv rather than
# hardcoded, so it always matches whatever data the currently-loaded
# model was actually trained on (see load_population_reference()).
_POPULATION_REFERENCE_CACHE = None


def load_population_reference(csv_path: str = None) -> dict:
    global _POPULATION_REFERENCE_CACHE
    if _POPULATION_REFERENCE_CACHE is not None:
        return _POPULATION_REFERENCE_CACHE

    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "cognitive_state_dataset.csv"
        )
    import csv as csv_module

    sums = {c: 0.0 for c in FEATURE_COLS}
    sumsqs = {c: 0.0 for c in FEATURE_COLS}
    n = 0
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            n += 1
            for c in FEATURE_COLS:
                v = float(row[c])
                sums[c] += v
                sumsqs[c] += v * v

    ref = {}
    for c in FEATURE_COLS:
        mean = sums[c] / n
        var = max(sumsqs[c] / n - mean * mean, 1e-9)
        ref[c] = {"mean": mean, "std": math.sqrt(var)}

    _POPULATION_REFERENCE_CACHE = ref
    return ref


class PersonalCalibrator:
    """
    One instance per user (persisted to a single JSON file, not
    per-session). Call `update(raw_features)` on every window while
    `is_calibrated()` is False to accumulate baseline samples; call
    `transform(raw_features)` on every window (calibrated or not --
    it's a no-op / identity before enough samples exist) to get the
    population-scale-adjusted features to actually feed the predictor.
    """

    def __init__(self, profile_path: str, min_samples: int = 30,
                 max_zscore_clip: float = 4.0, population_reference: dict = None):
        self.profile_path = profile_path
        self.min_samples = min_samples
        self.max_zscore_clip = max_zscore_clip
        self.population_reference = population_reference or load_population_reference()

        # Welford's online algorithm state per feature: running count,
        # mean, and M2 (sum of squared deviations from the running mean).
        # Chosen over storing raw samples because it's O(1) memory/update
        # and numerically stable, and because it merges cleanly across
        # process restarts (see _load/_save below) without needing to
        # keep every individual sample around.
        self._n = {c: 0 for c in FEATURE_COLS}
        self._mean = {c: 0.0 for c in FEATURE_COLS}
        self._m2 = {c: 0.0 for c in FEATURE_COLS}
        self._load()

    def _load(self):
        if not os.path.exists(self.profile_path):
            return
        try:
            with open(self.profile_path) as f:
                data = json.load(f)
            for c in FEATURE_COLS:
                if c in data:
                    self._n[c] = data[c]["n"]
                    self._mean[c] = data[c]["mean"]
                    self._m2[c] = data[c]["m2"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # corrupt/partial profile -> start fresh rather than crash

    def _save(self):
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        data = {
            c: {"n": self._n[c], "mean": self._mean[c], "m2": self._m2[c]}
            for c in FEATURE_COLS
        }
        tmp_path = self.profile_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.profile_path)  # atomic, avoids a torn write on crash

    def update(self, raw_features: dict):
        """Feed one window's raw feature dict into the running baseline."""
        for c in FEATURE_COLS:
            if c not in raw_features:
                continue
            x = float(raw_features[c])
            self._n[c] += 1
            delta = x - self._mean[c]
            self._mean[c] += delta / self._n[c]
            delta2 = x - self._mean[c]
            self._m2[c] += delta * delta2
        self._save()

    def user_stats(self, feature: str):
        n = self._n[feature]
        if n < 2:
            return None
        # NOTE: no artificial floor on variance here (unlike
        # load_population_reference()'s 1e-9 floor, which exists only to
        # avoid a math domain error on population data that's never truly
        # constant). A user CAN have truly-zero variance for a window
        # (e.g. 0 backspaces every single window) -- std must be allowed
        # to come out as exactly 0.0 so transform()'s `user_std < 1e-6`
        # check below actually catches it and takes the mean-shift
        # fallback, instead of computing a std of ~3e-5 that then
        # produces a several-hundred-sigma z-score silently clipped to
        # the cap (a real bug caught by this project's own test suite).
        var = max(self._m2[feature] / (n - 1), 0.0)
        return {"n": n, "mean": self._mean[feature], "std": math.sqrt(var)}

    def is_calibrated(self, feature: str = None) -> bool:
        if feature is not None:
            return self._n[feature] >= self.min_samples
        return min(self._n.values()) >= self.min_samples

    def calibration_progress(self) -> float:
        """0.0-1.0, based on the slowest-calibrated feature (some
        features naturally get fewer valid samples per window than
        others, e.g. backspace_rate needs >=1 keystroke in the window)."""
        return min(1.0, min(self._n.values()) / self.min_samples)

    def transform(self, raw_features: dict) -> dict:
        """
        Returns a NEW dict, population-scale-adjusted per feature. Safe
        to call before calibration finishes -- returns features
        unchanged (identity) for any feature that isn't calibrated yet,
        so this is always safe to call unconditionally rather than
        needing an is_calibrated() check at every call site.
        """
        out = dict(raw_features)
        for c in FEATURE_COLS:
            if c not in raw_features:
                continue
            if not self.is_calibrated(c):
                continue  # not enough data for this feature yet -- pass through raw

            stats = self.user_stats(c)
            ref = self.population_reference[c]
            user_std = stats["std"]

            if user_std < 1e-6:
                # This feature was ~constant for this user during
                # calibration (e.g. truly 0 backspaces every window) --
                # a near-zero std would blow up the ratio below into an
                # extreme value from a tiny raw difference. Fall back to
                # a pure mean-shift (no scale correction) instead.
                out[c] = ref["mean"] + (raw_features[c] - stats["mean"])
                continue

            z = (raw_features[c] - stats["mean"]) / user_std
            z = max(-self.max_zscore_clip, min(self.max_zscore_clip, z))
            out[c] = ref["mean"] + z * ref["std"]
        return out

    def transform_sequence(self, sequence, feature_cols):
        """Same transform, applied row-wise to a (T, 14) sequence for the
        CNN predictor -- each timestep gets the same per-feature
        adjustment as the single-row RF path."""
        import numpy as np
        out = np.array(sequence, dtype=np.float32, copy=True)
        for j, c in enumerate(feature_cols):
            if c not in FEATURE_COLS or not self.is_calibrated(c):
                continue
            stats = self.user_stats(c)
            ref = self.population_reference[c]
            user_std = stats["std"]
            if user_std < 1e-6:
                out[:, j] = ref["mean"] + (sequence[:, j] - stats["mean"])
                continue
            z = (sequence[:, j] - stats["mean"]) / user_std
            z = np.clip(z, -self.max_zscore_clip, self.max_zscore_clip)
            out[:, j] = ref["mean"] + z * ref["std"]
        return out
