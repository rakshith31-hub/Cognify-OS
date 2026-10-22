"""
train_model.py
---------------
Trains a RandomForestClassifier to predict cognitive_state from windowed
eye-tracking + mouse + keyboard features, evaluates it, and saves:
  - models/cognitive_state_model.joblib
  - models/scaler.joblib
  - models/label_encoder.joblib
  - outputs/classification_report.txt
  - outputs/confusion_matrix.png
  - outputs/feature_importance.png
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "cognitive_state_dataset.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")

FEATURE_COLS = [
    "blink_rate_per_min", "avg_ear", "fixation_duration_ms", "gaze_dispersion",
    "saccade_rate_per_min", "pupil_diameter_norm",
    "mouse_speed_mean", "mouse_speed_var", "click_rate_per_min", "mouse_idle_ratio",
    "typing_speed_kpm", "key_interval_var", "backspace_rate", "long_pause_count",
]
LABEL_COL = "cognitive_state"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y_raw = df[LABEL_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Classes:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # 5-fold cross validation on training set
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Held-out test accuracy: {test_acc:.4f}")

    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=3)
    print(report)

    # mean confidence (max predicted probability) for correct vs incorrect predictions
    max_proba = y_proba.max(axis=1)
    correct_mask = (y_pred == y_test)
    print(f"Mean confidence on CORRECT predictions:   {max_proba[correct_mask].mean():.3f}")
    print(f"Mean confidence on INCORRECT predictions: {max_proba[~correct_mask].mean():.3f}")

    # ---- save artifacts ----
    joblib.dump(clf, f"{MODELS_DIR}/cognitive_state_model.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.joblib")
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    with open(f"{OUTPUTS_DIR}/classification_report.txt", "w") as f:
        f.write(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")
        f.write(f"Held-out test accuracy: {test_acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nMean confidence on CORRECT predictions:   {max_proba[correct_mask].mean():.3f}\n")
        f.write(f"Mean confidence on INCORRECT predictions: {max_proba[~correct_mask].mean():.3f}\n")

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (test acc={test_acc:.3f})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # feature importance plot
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[order], y=np.array(FEATURE_COLS)[order], color="steelblue")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/feature_importance.png", dpi=150)
    plt.close()

    print("\nSaved model, scaler, label encoder to models/")
    print("Saved evaluation report + plots to outputs/")


if __name__ == "__main__":
    main()
