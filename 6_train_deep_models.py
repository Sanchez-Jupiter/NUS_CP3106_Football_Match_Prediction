"""
Step 6: Deep learning training pipeline (PyTorch)

This script trains two neural-network classifiers:
1) Pre-match final result prediction
2) In-play final result prediction

Main design choices:
- Temporal split by date (last 20% as test)
- Fixture-level group holdout for in-play data (avoids leakage across minutes)
- MLP with BatchNorm + Dropout
- Class-weighted CrossEntropyLoss to reduce class imbalance impact
- Early stopping on validation macro-F1

Outputs:
- models/pretrain_model_deep.pt
- models/inplay_model_deep.pt
- reports/pretrain_report_deep.txt
- reports/inplay_report_deep.txt
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Windows scientific stacks can load multiple OpenMP runtimes via PyTorch/NumPy/sklearn.
# This guard allows the script to start instead of failing on duplicate libiomp initialization.
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LABEL_ORDER = ["A", "D", "H"]

# set random seeds for reproducibility
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability(0)
        device_tag = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and device_tag not in supported_arches:
            print(
                f"CUDA device architecture {device_tag} is not supported by the installed PyTorch build; "
                "falling back to CPU."
            )
            return torch.device("cpu")
    except Exception as exc:
        print(f"CUDA capability check failed ({exc}); falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cuda")

# Multi-layer Perceptron classifier with BatchNorm and Dropout
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        hidden1 = min(256, max(64, input_dim * 2))
        hidden2 = min(128, max(32, input_dim))

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 16


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _select_numeric_features(df: pd.DataFrame, exclude_cols: set[str]) -> List[str]:
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    return [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]


def _temporal_holdout(
    df: pd.DataFrame,
    date_col: str,
    test_frac: float = 0.2,
    group_col: str | None = None,
) -> Tuple[pd.Series, pd.Series]:
    temp = df.copy()
    temp[date_col] = _to_datetime(temp[date_col])

    if group_col is None:
        temp = temp.sort_values(date_col)
        n_test = max(1, int(len(temp) * test_frac))
        test_idx = temp.tail(n_test).index
        test_mask = df.index.isin(test_idx)
        train_mask = ~test_mask
        return pd.Series(train_mask, index=df.index), pd.Series(test_mask, index=df.index)

    group_df = (
        temp[[group_col, date_col]]
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[group_col], keep="first")
    )
    n_test_groups = max(1, int(len(group_df) * test_frac))
    test_groups = set(group_df.tail(n_test_groups)[group_col].tolist())

    test_mask = df[group_col].isin(test_groups)
    train_mask = ~test_mask
    return train_mask, test_mask

# Split indices for training and validation sets
def _train_val_split_indices(n: int, val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    split = int(n * (1.0 - val_frac))
    return idx[:split], idx[split:]

# Standardize features based on training set statistics
def _standardize_fit_transform(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    X_test_s = (X_test - mean) / std
    return X_train_s, X_val_s, X_test_s, mean, std

def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _macro_f1_from_logits(logits: torch.Tensor, y_true: np.ndarray) -> float:
    pred = logits.argmax(dim=1).cpu().numpy()
    return float(f1_score(y_true, pred, average="macro"))


def _evaluate_probs(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def train_one_task(
    task_name: str,
    data_file: Path,
    exclude_cols: set[str],
    model_path: Path,
    report_path: Path,
    group_col: str | None,
    config: TrainConfig,
) -> None:
    print("\n" + "=" * 72)
    print(f"Deep training for: {task_name}")
    print("=" * 72)

    df = pd.read_csv(data_file)
    feature_cols = _select_numeric_features(df, exclude_cols)

    X_df = df[feature_cols].fillna(0)
    y_raw = df["result"].astype(str)

    label_to_idx = {label: i for i, label in enumerate(LABEL_ORDER)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    y = y_raw.map(label_to_idx).values

    train_mask, test_mask = _temporal_holdout(df, date_col="date", test_frac=0.2, group_col=group_col)
    X_train_df, X_test_df = X_df[train_mask], X_df[test_mask]
    y_train_all, y_test = y[train_mask], y[test_mask]

    # Keep training set in temporal order and split last 15% for validation
    X_train_np = X_train_df.values.astype(np.float32)
    y_train_np = y_train_all.astype(np.int64)

    tr_idx, val_idx = _train_val_split_indices(len(X_train_np), val_frac=0.15)
    X_tr, y_tr = X_train_np[tr_idx], y_train_np[tr_idx]
    X_val, y_val = X_train_np[val_idx], y_train_np[val_idx]
    X_test = X_test_df.values.astype(np.float32)

    X_tr, X_val, X_test, mean, std = _standardize_fit_transform(X_tr, X_val, X_test)

    device = _select_device()
    print(f"Using device: {device}")
    print(f"Samples - train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_test)}")

    train_loader = _to_loader(X_tr, y_tr, config.batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, config.batch_size, shuffle=False)

    model = MLPClassifier(input_dim=X_tr.shape[1], num_classes=3).to(device)

    # Class weights from training split only
    class_counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            train_loss += float(loss.item()) * batch_size
            n_train += batch_size

        train_loss = train_loss / max(n_train, 1)

        model.eval()
        val_logits_list = []
        y_val_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_logits_list.append(logits.cpu())
                y_val_list.append(yb)

        val_logits = torch.cat(val_logits_list, dim=0)
        y_val_np = torch.cat(y_val_list, dim=0).numpy()
        val_f1 = _macro_f1_from_logits(val_logits, y_val_np)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{config.epochs} - train_loss={train_loss:.4f}, val_f1_macro={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    probs = _evaluate_probs(model, X_test, device)
    pred_idx = probs.argmax(axis=1)

    y_test_labels = np.array([idx_to_label[i] for i in y_test])
    pred_labels = np.array([idx_to_label[i] for i in pred_idx])

    acc = accuracy_score(y_test_labels, pred_labels)
    f1m = f1_score(y_test_labels, pred_labels, average="macro")

    try:
        from sklearn.preprocessing import label_binarize

        y_test_bin = label_binarize(y_test_labels, classes=LABEL_ORDER)
        auc = roc_auc_score(y_test_bin, probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    cls_report = classification_report(y_test_labels, pred_labels, target_names=["Away Win", "Draw", "Home Win"])
    cm = confusion_matrix(y_test_labels, pred_labels, labels=LABEL_ORDER)

    minute_text = ""
    if task_name.lower().startswith("in-play") and "minute" in df.columns:
        eval_df = df.loc[test_mask].copy()
        eval_df["pred"] = pred_labels
        minute_acc = {}
        for minute, group in eval_df.groupby("minute"):
            minute_acc[minute] = accuracy_score(group["result"], group["pred"])
        minute_text += "Accuracy by minute (test holdout):\n"
        for minute in sorted(minute_acc.keys()):
            macc = minute_acc[minute]
            minute_text += f"  {int(minute):2d}': {macc:.4f}\n"
    # This per-minute breakdown can provide insights into how the model's predictive 
    # performance evolves as the match progresses, which is especially relevant for in-play models.
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": X_tr.shape[1],
        "num_classes": 3,
        "features": feature_cols,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "norm_mean": mean,
        "norm_std": std,
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "roc_auc_weighted_ovr": float(auc),
            "best_val_f1": float(best_val_f1),
            "best_epoch": int(best_epoch + 1),
        },
        "split_strategy": "temporal last 20% by date",
        "group_holdout": group_col,
        "config": config.__dict__,
    }
    torch.save(checkpoint, model_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"DEEP LEARNING REPORT - {task_name}\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Feature count: {len(feature_cols)}\n")
        f.write("Split strategy: temporal last 20% by date\n")
        if group_col:
            f.write(f"Group holdout column: {group_col}\n")
        f.write(f"Train/Val/Test: {len(X_tr)}/{len(X_val)}/{len(X_test)}\n\n")

        f.write("Model config:\n")
        f.write(str(config.__dict__) + "\n\n")

        f.write(f"Best epoch: {best_epoch + 1}\n")
        f.write(f"Best val macro-F1: {best_val_f1:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Macro-F1: {f1m:.4f}\n")
        f.write(f"Test ROC-AUC (weighted ovr): {auc:.4f}\n\n")

        f.write("Classification report:\n")
        f.write(cls_report + "\n")

        f.write("Confusion matrix [A, D, H]:\n")
        f.write(str(cm) + "\n\n")

        if minute_text:
            f.write(minute_text + "\n")

    print("\n" + "-" * 72)
    print(f"Best epoch: {best_epoch + 1} | Best val F1-macro: {best_val_f1:.4f}")
    print(f"Test metrics -> Accuracy: {acc:.4f}, F1-macro: {f1m:.4f}, ROC-AUC(w-ovr): {auc:.4f}")
    print(f"Model saved:  {model_path}")
    print(f"Report saved: {report_path}")


def main() -> None:
    set_seed(SEED)
    cfg = TrainConfig(epochs=500, batch_size=128, lr=1e-3, weight_decay=1e-4, patience=16)

    train_one_task(
        task_name="Pre-match",
        data_file=DATA_DIR / "pretrain_dataset.csv",
        exclude_cols={
            "fixture_id",
            "date",
            "home_team",
            "away_team",
            "result",
            "goals_home",
            "goals_away",
        },
        model_path=MODEL_DIR / "pretrain_model_deep.pt",
        report_path=REPORT_DIR / "pretrain_report_deep.txt",
        group_col=None,
        config=cfg,
    )

    train_one_task(
        task_name="In-play",
        data_file=DATA_DIR / "inplay_dataset.csv",
        exclude_cols={
            "fixture_id",
            "date",
            "home_team",
            "away_team",
            "result",
            "ft_home",
            "ft_away",
        },
        model_path=MODEL_DIR / "inplay_model_deep.pt",
        report_path=REPORT_DIR / "inplay_report_deep.txt",
        group_col="fixture_id",
        config=cfg,
    )

    print("\n" + "=" * 72)
    print("[OK] Deep learning pipeline complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()
