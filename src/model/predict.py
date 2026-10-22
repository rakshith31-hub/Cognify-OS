"""
predict.py
----------
Loads the trained model/scaler/label-encoder and exposes predictor classes
used by the real-time pipeline (and usable standalone for batch predictions).

Two predictors, same input/output contract ({state, confidence,
all_probabilities}), different model + input shape:
  - CognitiveStatePredictor    -> RandomForest, needs ONE aggregated
                                   14-feature dict (FeatureAggregator.compute_features()).
  - CNNCognitiveStatePredictor -> 1D-CNN, needs a (T, 14) SEQUENCE
                                   (FeatureAggregator.compute_sequence_features()).
torch is only imported inside CNNCognitiveStatePredictor so environments
that only use the RF model don't need it installed.
"""

import json
import os
import joblib
import numpy as np

# Resolve relative to the project root (two levels up from this file:
# src/model/predict.py -> src/model -> src -> project root), so the
# predictor works no matter what directory the caller was launched from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


class CognitiveStatePredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.model = joblib.load(f"{models_dir}/cognitive_state_model.joblib")
        self.scaler = joblib.load(f"{models_dir}/scaler.joblib")
        self.label_encoder = joblib.load(f"{models_dir}/label_encoder.joblib")
        with open(f"{models_dir}/feature_columns.json") as f:
            self.feature_columns = json.load(f)

    def predict(self, feature_dict: dict) -> dict:
        """
        feature_dict: {feature_name: value, ...} for ALL feature_columns.
        Returns: {
            "state": str,
            "confidence": float,                # confidence of the winning class
            "all_probabilities": {state: prob},  # full distribution
        }
        """
        x = np.array([[feature_dict[c] for c in self.feature_columns]])
        x_scaled = self.scaler.transform(x)
        proba = self.model.predict_proba(x_scaled)[0]
        pred_idx = int(np.argmax(proba))
        state = self.label_encoder.inverse_transform([pred_idx])[0]

        all_probs = {
            cls: float(p) for cls, p in zip(self.label_encoder.classes_, proba)
        }
        return {
            "state": state,
            "confidence": float(proba[pred_idx]),
            "all_probabilities": all_probs,
        }


class CNNCognitiveStatePredictor:
    """1D-CNN counterpart to CognitiveStatePredictor. Same output contract,
    but predict() takes a (T, 14) sequence (T timesteps, 14 features in
    feature_columns order) instead of a single aggregated feature dict --
    use FeatureAggregator.compute_sequence_features() to build it."""

    def __init__(self, models_dir: str = MODELS_DIR):
        import torch  # local import: only needed if this class is used
        from src.model.train_cnn import CognitiveStateCNN

        self._torch = torch
        checkpoint = torch.load(
            f"{models_dir}/cognitive_state_cnn.pt", map_location="cpu", weights_only=False
        )
        self.model = CognitiveStateCNN(
            checkpoint["n_features"], checkpoint["n_classes"], checkpoint["t_steps"]
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.t_steps = checkpoint["t_steps"]

        with open(f"{models_dir}/cnn_scaler.json") as f:
            scaler = json.load(f)
        self.mean = np.array(scaler["mean"], dtype=np.float32)
        self.std = np.array(scaler["std"], dtype=np.float32)

        self.label_encoder = joblib.load(f"{models_dir}/label_encoder.joblib")
        with open(f"{models_dir}/feature_columns.json") as f:
            self.feature_columns = json.load(f)

    def predict(self, sequence: np.ndarray) -> dict:
        """
        sequence: (T, 14) array, columns matching self.feature_columns order,
        rows = timesteps (oldest -> newest).
        """
        torch = self._torch
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.shape != (self.t_steps, len(self.feature_columns)):
            raise ValueError(
                f"expected sequence shape ({self.t_steps}, {len(self.feature_columns)}), got {seq.shape}"
            )
        seq_norm = (seq - self.mean) / self.std
        x = torch.as_tensor(seq_norm, dtype=torch.float32).unsqueeze(0)  # (1, T, C)

        with torch.no_grad():
            logits = self.model(x)
            proba = torch.softmax(logits, dim=1).squeeze(0).numpy()

        pred_idx = int(np.argmax(proba))
        state = self.label_encoder.inverse_transform([pred_idx])[0]
        all_probs = {cls: float(p) for cls, p in zip(self.label_encoder.classes_, proba)}
        return {
            "state": state,
            "confidence": float(proba[pred_idx]),
            "all_probabilities": all_probs,
        }


class EnsembleCognitiveStatePredictor:
    """
    Combines RandomForest + CNN via weighted probability averaging.
    Default cnn_weight=0.2 (80/20 RF/CNN) is empirically grounded, not
    arbitrary -- see src/model/evaluate_ensemble.py, which found this
    weight range consistently outperformed BOTH models alone on the
    harder, more realistic live-pipeline test, across two independent
    large-sample runs (94.2% and 92.9% vs RF-alone's 90.0%/82.1% and
    CNN-alone's 74.2%/74.6%), while tying both on in-distribution data.
    Re-run that script if the underlying models are ever retrained --
    the optimal weight isn't guaranteed to stay the same.

    predict() needs BOTH input shapes since RF and CNN need different
    ones -- call FeatureAggregator.compute_features() AND
    compute_sequence_features() on the same window and pass both here,
    rather than this class reaching into the aggregator itself (keeps
    the predictor/aggregator boundary the same as the other two classes).
    """

    def __init__(self, models_dir: str = MODELS_DIR, cnn_weight: float = 0.2):
        if not (0.0 <= cnn_weight <= 1.0):
            raise ValueError(f"cnn_weight must be in [0, 1], got {cnn_weight}")
        self.cnn_weight = cnn_weight
        self.rf = CognitiveStatePredictor(models_dir=models_dir)
        self.cnn = CNNCognitiveStatePredictor(models_dir=models_dir)
        self.t_steps = self.cnn.t_steps  # so callers know what sequence length to build

    def predict(self, feature_dict: dict, sequence: np.ndarray) -> dict:
        rf_out = self.rf.predict(feature_dict)
        cnn_out = self.cnn.predict(sequence)

        classes = rf_out["all_probabilities"].keys()
        combined = {
            c: (1 - self.cnn_weight) * rf_out["all_probabilities"][c]
               + self.cnn_weight * cnn_out["all_probabilities"][c]
            for c in classes
        }
        state = max(combined, key=combined.get)
        return {
            "state": state,
            "confidence": combined[state],
            "all_probabilities": combined,
            # kept for transparency/debugging -- not part of the contract
            # the other two predictors expose, callers that only expect
            # {state, confidence, all_probabilities} are unaffected
            "component_predictions": {"rf": rf_out, "cnn": cnn_out},
        }


if __name__ == "__main__":
    # quick smoke test with a hand-crafted "focus-like" feature vector
    predictor = CognitiveStatePredictor()
    sample = {
        "blink_rate_per_min": 16,
        "avg_ear": 0.29,
        "fixation_duration_ms": 360,
        "gaze_dispersion": 0.045,
        "saccade_rate_per_min": 26,
        "pupil_diameter_norm": 0.56,
        "mouse_speed_mean": 520,
        "mouse_speed_var": 4500,
        "click_rate_per_min": 7.5,
        "mouse_idle_ratio": 0.20,
        "typing_speed_kpm": 210,
        "key_interval_var": 7500,
        "backspace_rate": 0.06,
        "long_pause_count": 1.0,
    }
    result = predictor.predict(sample)
    print(json.dumps(result, indent=2))
