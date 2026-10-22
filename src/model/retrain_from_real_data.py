"""
retrain_from_real_data.py
---------------------------
Retrains the RandomForest classifier on SYNTHETIC + REAL data combined --
the actual mechanism that closes the gap described in this project's
README ("the classifier is trained entirely on synthetic data and never
updates from real usage"). Triggered by `cognitive-monitor retrain`, or
automatically at daemon startup if auto_retrain_on_startup is enabled in
config (see daemon.py).

WHAT THIS DOES, PRECISELY (so it isn't oversold):
  1. Loads data/cognitive_state_dataset.csv (synthetic, thousands of rows)
     and logs/real_labels.csv (real, human-labeled via `cognitive-monitor
     label <state>` -- almost certainly far fewer rows).
  2. Concatenates them, but weights REAL rows more heavily in training
     (sample_weight), proportional to how outnumbered they are by
     synthetic rows -- otherwise a few dozen real labels get statistically
     drowned out by thousands of synthetic ones and change nothing.
  3. Evaluates on a REAL-only held-out split, separately from the usual
     synthetic-based accuracy number -- this is the number that actually
     answers "does this work on YOU", which the original 88.5%/95.8%
     numbers never could (see README).
  4. Refuses to silently deploy a worse model: compares the new real-
     holdout accuracy against the last successful retrain's (tracked in
     models/retrain_history.json) and requires --force to proceed if it
     regressed by more than --max-regression (default 10 percentage
     points). The OLD model files are backed up before any overwrite
     either way.

WHAT THIS DOES NOT DO:
  - Retrain the CNN (src/model/train_cnn.py) -- that needs 10-timestep
    SEQUENCES per label, which `cognitive-monitor label` doesn't collect
    (it captures one aggregated window, matching what compute_features()
    gives the RF path). Extending real-label collection to sequences
    would be a separate, larger piece of work.
  - Run itself on a schedule. It's invoked explicitly (CLI command) or
    once at daemon startup if opted into -- never mid-session.
"""

import argparse
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYNTHETIC_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "cognitive_state_dataset.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")
HISTORY_PATH = os.path.join(MODELS_DIR, "retrain_history.json")

FEATURE_COLS = [
    "blink_rate_per_min", "avg_ear", "fixation_duration_ms", "gaze_dispersion",
    "saccade_rate_per_min", "pupil_diameter_norm",
    "mouse_speed_mean", "mouse_speed_var", "click_rate_per_min", "mouse_idle_ratio",
    "typing_speed_kpm", "key_interval_var", "backspace_rate", "long_pause_count",
]
LABEL_COL = "cognitive_state"
MIN_SAMPLES_PER_CLASS = 8      # below this, a class's real data is too thin to trust at all
MIN_TOTAL_REAL_SAMPLES = 30    # below this, don't bother -- not enough to move the needle
MAX_REAL_WEIGHT_MULTIPLIER = 5.0  # cap how much any single real row can outweigh a synthetic one


def _load_history(history_path: str) -> dict:
    if os.path.exists(history_path):
        with open(history_path) as f:
            return json.load(f)
    return {"retrains": []}


def _save_history(history: dict, history_path: str):
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def _backup_current_model(models_dir: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(models_dir, "backups", ts)
    os.makedirs(backup_dir, exist_ok=True)
    for fname in ["cognitive_state_model.joblib", "scaler.joblib", "label_encoder.joblib"]:
        src = os.path.join(models_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, fname))
    return backup_dir


def retrain(log_dir: str, force: bool = False, max_regression: float = 0.10,
            min_total_real: int = MIN_TOTAL_REAL_SAMPLES, models_dir: str = None,
            outputs_dir: str = None) -> dict:
    """
    Returns a result dict: {"status": "ok"|"insufficient_data"|"regressed",
    ...details}. Never raises on "not enough data" or "would regress" --
    those are expected, normal outcomes, not errors, and the caller (CLI)
    is expected to print result["message"] either way.

    models_dir/outputs_dir default to the real project paths -- pass
    overrides (as this module's own tests do) to avoid ever touching the
    actual shipped model files during testing.
    """
    models_dir = models_dir or MODELS_DIR
    outputs_dir = outputs_dir or OUTPUTS_DIR
    history_path = os.path.join(models_dir, "retrain_history.json")
    real_path = os.path.join(log_dir, "real_labels.csv")
    if not os.path.exists(real_path):
        return {"status": "insufficient_data",
                "message": f"No real labels collected yet at {real_path}. "
                            f"Use `cognitive-monitor label <state>` to start collecting them."}

    real_df = pd.read_csv(real_path)
    real_df = real_df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    counts = real_df[LABEL_COL].value_counts().to_dict()
    total_real = len(real_df)
    classes_with_enough = [c for c in ["focus", "fatigue", "overload", "distracted"]
                            if counts.get(c, 0) >= MIN_SAMPLES_PER_CLASS]

    if total_real < min_total_real or len(classes_with_enough) < 2:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {total_real} real labels collected so far ({counts}), need at least "
                f"{min_total_real} total AND {MIN_SAMPLES_PER_CLASS}+ in at least 2 classes. "
                f"Keep using `cognitive-monitor label <state>` -- this is expected early on, "
                f"not an error."
            ),
            "counts": counts, "total_real": total_real,
        }

    synthetic_df = pd.read_csv(SYNTHETIC_DATA_PATH)

    # ---- real-only holdout, evaluated SEPARATELY from synthetic accuracy ----
    # stratify=None if any class has <2 real samples in the split (train_test_split
    # would error on stratifying a class with only 1 member)
    real_class_counts = real_df[LABEL_COL].value_counts()
    can_stratify = (real_class_counts >= 2).all()
    real_train, real_test = train_test_split(
        real_df, test_size=0.25, random_state=42,
        stratify=real_df[LABEL_COL] if can_stratify else None,
    )

    train_df = pd.concat([synthetic_df, real_train], ignore_index=True)
    X_train_raw = train_df[FEATURE_COLS].values
    y_train_raw = train_df[LABEL_COL].values

    le = LabelEncoder()
    le.fit(sorted(set(y_train_raw) | set(real_test[LABEL_COL])))
    y_train = le.transform(y_train_raw)

    # ---- sample weights: real rows outweigh synthetic ones proportional to scarcity ----
    n_synthetic, n_real_train = len(synthetic_df), len(real_train)
    real_weight = min(n_synthetic / max(n_real_train, 1), MAX_REAL_WEIGHT_MULTIPLIER)
    sample_weight = np.concatenate([
        np.ones(n_synthetic),
        np.full(n_real_train, real_weight),
    ])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    X_test_real = scaler.transform(real_test[FEATURE_COLS].values)
    y_test_real = le.transform(real_test[LABEL_COL].values)
    y_pred_real = clf.predict(X_test_real)
    real_holdout_acc = accuracy_score(y_test_real, y_pred_real)
    real_report = classification_report(
        y_test_real, y_pred_real, labels=le.transform(le.classes_),
        target_names=le.classes_, digits=3, zero_division=0,
    )

    history = _load_history(history_path)
    last_real_acc = history["retrains"][-1]["real_holdout_accuracy"] if history["retrains"] else None
    regressed = last_real_acc is not None and (last_real_acc - real_holdout_acc) > max_regression

    result = {
        "status": "regressed" if (regressed and not force) else "ok",
        "real_holdout_accuracy": real_holdout_acc,
        "real_holdout_report": real_report,
        "n_real_total": total_real, "n_real_train": n_real_train, "n_real_test": len(real_test),
        "n_synthetic": n_synthetic, "real_sample_weight": real_weight,
        "previous_real_holdout_accuracy": last_real_acc,
        "counts": counts,
    }

    if regressed and not force:
        result["message"] = (
            f"New model's real-holdout accuracy ({real_holdout_acc:.1%}) is "
            f"{(last_real_acc - real_holdout_acc):.1%} WORSE than the last retrain's "
            f"({last_real_acc:.1%}) -- more than the {max_regression:.0%} regression "
            f"threshold. NOT deploying it. Re-run with --force if you want to deploy "
            f"anyway (e.g. you know the extra real labels are trustworthy and the "
            f"previous number was a fluke on a small holdout)."
        )
        return result

    # ---- deploy: back up old model, then overwrite ----
    os.makedirs(models_dir, exist_ok=True)
    backup_dir = _backup_current_model(models_dir)
    joblib.dump(clf, os.path.join(models_dir, "cognitive_state_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))

    history["retrains"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "real_holdout_accuracy": real_holdout_acc,
        "n_real_total": total_real, "n_real_train": n_real_train, "n_real_test": len(real_test),
        "n_synthetic": n_synthetic, "real_sample_weight": real_weight,
        "backup_dir": backup_dir,
    })
    _save_history(history, history_path)

    os.makedirs(outputs_dir, exist_ok=True)
    with open(os.path.join(outputs_dir, "real_holdout_report.txt"), "w") as f:
        f.write(f"Real-holdout accuracy: {real_holdout_acc:.4f}\n")
        f.write(f"({len(real_test)} real test samples, {n_real_train} real train samples "
                f"weighted {real_weight:.2f}x each against {n_synthetic} synthetic samples)\n\n")
        f.write(real_report)

    result["backup_dir"] = backup_dir
    result["message"] = (
        f"Retrained and deployed. Real-holdout accuracy: {real_holdout_acc:.1%} "
        f"(on {len(real_test)} real held-out samples -- separate from, and more meaningful "
        f"than, the synthetic-data numbers in the README). Previous model backed up to "
        f"{backup_dir}."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=os.path.join(_PROJECT_ROOT, "logs"))
    parser.add_argument("--force", action="store_true",
                         help="Deploy even if real-holdout accuracy regressed vs. last retrain")
    parser.add_argument("--max-regression", type=float, default=0.10,
                         help="Max allowed accuracy drop (as a fraction, e.g. 0.10 = 10pp) before refusing")
    args = parser.parse_args()

    result = retrain(args.log_dir, force=args.force, max_regression=args.max_regression)
    print(result["message"])
    if result["status"] == "ok" and "real_holdout_report" in result:
        print()
        print(result["real_holdout_report"])
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
