"""
train_cnn.py
------------
Trains a 1D-CNN on the RAW PER-SECOND (10 x 14) feature sequences, instead
of the single aggregated row RandomForest uses. Convolving over the time
axis lets the model learn temporal patterns (e.g. a backspace rate that's
rising through the window vs one that's flat) that the mean/var aggregation
in train_model.py discards.

Saves:
  - models/cognitive_state_cnn.pt        (PyTorch state_dict + arch config)
  - models/cnn_scaler.json               (per-feature mean/std for normalization)
  - models/label_encoder.joblib          (reused/shared with the RF model)
  - models/feature_columns.json          (reused/shared with the RF model)
  - outputs/cnn_classification_report.txt
  - outputs/cnn_confusion_matrix.png
  - outputs/cnn_training_curve.png
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "cognitive_state_sequences.npz")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------- model ----
class CognitiveStateCNN(nn.Module):
    """
    Input:  (batch, T=10, C=14)  -> transposed internally to (batch, C, T)
            for Conv1d, which convolves along the TIME axis per feature
            channel (so it can learn e.g. "backspace_rate trending up").
    """
    def __init__(self, n_features: int, n_classes: int, t_steps: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.act = nn.ReLU()

    def forward(self, x):  # x: (batch, T, C)
        x = x.transpose(1, 2)  # -> (batch, C, T)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        return self.fc2(x)  # logits


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def per_feature_normalize_fit(X_train):
    """Fit mean/std per feature channel over ALL timesteps of the training
    set (equivalent in spirit to StandardScaler, but shape-aware for
    (N, T, C) sequences instead of (N, C) rows)."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


def train_one_model(X_train, y_train, X_val, y_val, n_features, n_classes, t_steps,
                     epochs=EPOCHS, verbose=False):
    model = CognitiveStateCNN(n_features, n_classes, t_steps).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # class-balanced loss (mirrors class_weight="balanced" used for the RF)
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = torch.as_tensor(
        class_counts.sum() / (n_classes * np.maximum(class_counts, 1)), dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=256, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_val_acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = np.load(DATA_PATH, allow_pickle=True)
    X, y_raw = data["X"], data["y"]
    feature_cols = list(data["feature_cols"])
    n_features, t_steps = X.shape[2], X.shape[1]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print("Classes:", list(le.classes_))
    print("Sequence shape:", X.shape)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ---- 5-fold CV on the trainval split, same protocol as train_model.py ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_accs = []
    print("\nRunning 5-fold CV...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        X_tr, X_va = X_trainval[tr_idx], X_trainval[va_idx]
        y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]
        mean, std = per_feature_normalize_fit(X_tr)
        X_tr_n, X_va_n = normalize(X_tr, mean, std), normalize(X_va, mean, std)
        _, _, fold_acc = train_one_model(X_tr_n, y_tr, X_va_n, y_va, n_features, n_classes, t_steps, epochs=25)
        cv_accs.append(fold_acc)
        print(f"  fold {fold+1}: val_acc={fold_acc:.4f}")
    cv_accs = np.array(cv_accs)
    print(f"5-fold CV accuracy: {cv_accs.mean():.4f} +/- {cv_accs.std():.4f}")

    # ---- final model: train on full trainval, evaluate on held-out test ----
    print("\nTraining final model on full train/val split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=SEED, stratify=y_trainval
    )
    mean, std = per_feature_normalize_fit(X_train)
    X_train_n = normalize(X_train, mean, std)
    X_val_n = normalize(X_val, mean, std)
    X_test_n = normalize(X_test, mean, std)

    model, history, _ = train_one_model(
        X_train_n, y_train, X_val_n, y_val, n_features, n_classes, t_steps,
        epochs=EPOCHS, verbose=True
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_test_n, dtype=torch.float32).to(DEVICE))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = proba.argmax(axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out test accuracy: {test_acc:.4f}")
    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=3)
    print(report)

    max_proba = proba.max(axis=1)
    correct_mask = (y_pred == y_test)
    mean_conf_correct = max_proba[correct_mask].mean()
    mean_conf_incorrect = max_proba[~correct_mask].mean() if (~correct_mask).any() else float("nan")
    print(f"Mean confidence on CORRECT predictions:   {mean_conf_correct:.3f}")
    print(f"Mean confidence on INCORRECT predictions: {mean_conf_incorrect:.3f}")

    # ---- save artifacts ----
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "n_classes": n_classes,
        "t_steps": t_steps,
    }, os.path.join(MODELS_DIR, "cognitive_state_cnn.pt"))

    with open(os.path.join(MODELS_DIR, "cnn_scaler.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "t_steps": t_steps}, f)

    # label encoder + feature columns are the SAME as the RF model's --
    # both models predict the same 4 classes over the same 14 named features
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    with open(os.path.join(OUTPUTS_DIR, "cnn_classification_report.txt"), "w") as f:
        f.write(f"5-fold CV accuracy: {cv_accs.mean():.4f} +/- {cv_accs.std():.4f}\n")
        f.write(f"Held-out test accuracy: {test_acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nMean confidence on CORRECT predictions:   {mean_conf_correct:.3f}\n")
        f.write(f"Mean confidence on INCORRECT predictions: {mean_conf_incorrect:.3f}\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"CNN Confusion Matrix (test acc={test_acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "cnn_confusion_matrix.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("CNN Training Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "cnn_training_curve.png"), dpi=150)
    plt.close()

    print("\nSaved CNN model + scaler to models/")
    print("Saved evaluation report + plots to outputs/")


if __name__ == "__main__":
    main()
