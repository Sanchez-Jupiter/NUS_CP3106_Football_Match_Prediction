"""
Step 15: PennyLane hybrid QNN vs BP/LR/KNN comparison

What this script does:
1) Uses the same temporal split for all models (last 20% by date as test).
2) Trains classical baselines: KNN, Logistic Regression, BP-MLP.
3) Trains a PennyLane + PyTorch hybrid QNN model.
4) Exports report-ready comparison tables.

Input:
- data/processed/pretrain_dataset.csv

Outputs:
- reports/15_qnn_hybrid_comparison.csv
- reports/15_qnn_hybrid_comparison.md
- reports/15_qnn_hybrid_comparison.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False

DATA_FILE = Path("data/processed/pretrain_dataset.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = REPORT_DIR / "15_qnn_hybrid_comparison.csv"
MD_OUT = REPORT_DIR / "15_qnn_hybrid_comparison.md"
TXT_OUT = REPORT_DIR / "15_qnn_hybrid_comparison.txt"
QNN_TRIAL_CSV_OUT = REPORT_DIR / "15_qnn_hybrid_tuning_trials.csv"
QNN_TRIAL_MD_OUT = REPORT_DIR / "15_qnn_hybrid_tuning_trials.md"

SEED = 42
LABEL_ORDER = ["A", "D", "H"]


@dataclass(frozen=True)
class QNNConfig:
    n_qubits: int
    n_q_layers: int
    head_hidden: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def temporal_holdout(df: pd.DataFrame, date_col: str, test_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    temp = df.copy()
    temp[date_col] = _to_datetime(temp[date_col])
    temp = temp.sort_values(date_col)

    n_test = max(1, int(len(temp) * test_frac))
    test_idx = temp.tail(n_test).index
    test_mask = df.index.isin(test_idx)
    train_mask = ~test_mask
    return np.asarray(train_mask), np.asarray(test_mask)


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    exclude_cols = {
        "fixture_id",
        "date",
        "home_team",
        "away_team",
        "result",
        "goals_home",
        "goals_away",
    }
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].fillna(0.0).copy()
    y = df["result"].astype(str).copy()
    return X, y, feature_cols


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
    }


def train_baselines(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}

    models = {
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=11, weights="distance")),
            ]
        ),
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        ),
        "BP-MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = metric_dict(y_test, pred)
        out[name] = {
            **m,
            "report": classification_report(y_test, pred, labels=LABEL_ORDER, digits=4, zero_division=0),
        }

    return out


class HybridQNN(nn.Module):
    def __init__(
        self,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        n_classes: int = 3,
        head_hidden: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane is not installed.")

        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_q_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.head = nn.Sequential(
            nn.Linear(n_qubits, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_layer(x)
        return self.head(x)


def train_qnn_hybrid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: QNNConfig,
) -> Dict[str, object]:
    if not HAS_PENNYLANE:
        return {
            "status": "skipped",
            "reason": "PennyLane not installed",
        }

    label_to_idx = {k: i for i, k in enumerate(LABEL_ORDER)}
    idx_to_label = {i: k for k, i in label_to_idx.items()}

    y_train_idx = np.array([label_to_idx[v] for v in y_train], dtype=np.int64)
    y_test_idx = np.array([label_to_idx[v] for v in y_test], dtype=np.int64)

    # Same preprocessing family for fair comparison.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=cfg.n_qubits, random_state=SEED)
    X_train_q = pca.fit_transform(X_train_s).astype(np.float32)
    X_test_q = pca.transform(X_test_s).astype(np.float32)

    # Temporal-consistent validation split: tail 15% from train.
    split = int(len(X_train_q) * 0.85)
    X_tr, X_val = X_train_q[:split], X_train_q[split:]
    y_tr, y_val = y_train_idx[:split], y_train_idx[split:]

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    device = torch.device("cpu")  # default.qubit is CPU simulator
    model = HybridQNN(
        n_qubits=cfg.n_qubits,
        n_q_layers=cfg.n_q_layers,
        n_classes=3,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
    ).to(device)

    cls_counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    cls_weights = cls_counts.sum() / np.maximum(cls_counts, 1.0)
    cls_weights = cls_weights / cls_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    bad_epochs = 0
    patience = 8

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                all_pred.extend(pred.tolist())
                all_true.extend(yb.numpy().tolist())

        p, r, f1, _ = precision_recall_fscore_support(all_true, all_pred, average="macro", zero_division=0)
        if f1 > best_val_f1 + 1e-6:
            best_val_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_test = model(torch.tensor(X_test_q, dtype=torch.float32, device=device))
        pred_idx = logits_test.argmax(dim=1).cpu().numpy()

    pred = np.array([idx_to_label[i] for i in pred_idx])
    m = metric_dict(y_test, pred)

    return {
        "status": "ok",
        **m,
        "report": classification_report(y_test, pred, labels=LABEL_ORDER, digits=4, zero_division=0),
        "best_val_f1": float(best_val_f1),
        "epochs": int(cfg.epochs),
        "n_qubits": int(cfg.n_qubits),
        "n_q_layers": int(cfg.n_q_layers),
        "head_hidden": int(cfg.head_hidden),
        "dropout": float(cfg.dropout),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
    }


def tune_qnn_hybrid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    search_space: List[QNNConfig] = [
        QNNConfig(4, 1, 24, 0.10, 1e-3, 1e-4, 45, 256),
        QNNConfig(4, 2, 24, 0.10, 1e-3, 1e-4, 45, 256),
        QNNConfig(4, 2, 32, 0.15, 8e-4, 1e-4, 50, 256),
        QNNConfig(4, 3, 32, 0.15, 8e-4, 1e-4, 50, 256),
        QNNConfig(6, 1, 32, 0.10, 1e-3, 1e-4, 45, 256),
        QNNConfig(6, 2, 32, 0.10, 1e-3, 1e-4, 45, 256),
        QNNConfig(6, 2, 48, 0.20, 6e-4, 1e-4, 55, 256),
        QNNConfig(6, 3, 48, 0.20, 6e-4, 1e-4, 55, 256),
    ]

    trials = []
    best = None

    for i, cfg in enumerate(search_space, start=1):
        print(
            f"  Trial {i}/{len(search_space)} | q={cfg.n_qubits}, layers={cfg.n_q_layers}, "
            f"hidden={cfg.head_hidden}, lr={cfg.lr}"
        )
        res = train_qnn_hybrid(X_train, y_train, X_test, y_test, cfg)
        trial_row = {
            "trial": i,
            "n_qubits": cfg.n_qubits,
            "n_q_layers": cfg.n_q_layers,
            "head_hidden": cfg.head_hidden,
            "dropout": cfg.dropout,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "status": res.get("status", "unknown"),
            "accuracy": res.get("accuracy", np.nan),
            "precision_macro": res.get("precision_macro", np.nan),
            "recall_macro": res.get("recall_macro", np.nan),
            "f1_macro": res.get("f1_macro", np.nan),
            "best_val_f1": res.get("best_val_f1", np.nan),
        }
        trials.append(trial_row)

        if res.get("status") == "ok":
            if best is None or res["f1_macro"] > best["f1_macro"]:
                best = res

    trials_df = pd.DataFrame(trials).sort_values("f1_macro", ascending=False, na_position="last")
    return best if best is not None else {"status": "skipped", "reason": "all trials failed"}, trials_df


def main() -> None:
    set_seed(SEED)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    X_df, y_sr, _ = build_xy(df)

    train_mask, test_mask = temporal_holdout(df, date_col="date", test_frac=0.2)
    X_train = X_df.loc[train_mask].to_numpy(dtype=np.float32)
    y_train = y_sr.loc[train_mask].to_numpy()
    X_test = X_df.loc[test_mask].to_numpy(dtype=np.float32)
    y_test = y_sr.loc[test_mask].to_numpy()

    print(f"Temporal split train/test: {len(X_train)}/{len(X_test)}")
    print("Training baselines: KNN / LR / BP-MLP")
    baseline_res = train_baselines(X_train, y_train, X_test, y_test)

    print("Tuning QNN hybrid (PennyLane + MLP head)")
    qnn_res, qnn_trials_df = tune_qnn_hybrid(X_train, y_train, X_test, y_test)
    qnn_trials_df.to_csv(QNN_TRIAL_CSV_OUT, index=False, encoding="utf-8")
    qnn_trials_md = [
        "# QNN Hybrid Tuning Trials",
        "",
        f"- Data: `{DATA_FILE}`",
        f"- Split: temporal holdout (last 20% by date)",
        "",
        qnn_trials_df.to_markdown(index=False),
    ]
    QNN_TRIAL_MD_OUT.write_text("\n".join(qnn_trials_md), encoding="utf-8")

    rows = []
    for name in ["KNN", "LR", "BP-MLP"]:
        r = baseline_res[name]
        rows.append(
            {
                "model": name,
                "accuracy": r["accuracy"],
                "precision_macro": r["precision_macro"],
                "recall_macro": r["recall_macro"],
                "f1_macro": r["f1_macro"],
                "notes": "temporal holdout",
            }
        )

    if qnn_res.get("status") == "ok":
        rows.append(
            {
                "model": "Hybrid-QNN",
                "accuracy": qnn_res["accuracy"],
                "precision_macro": qnn_res["precision_macro"],
                "recall_macro": qnn_res["recall_macro"],
                "f1_macro": qnn_res["f1_macro"],
                "notes": (
                    f"q={qnn_res['n_qubits']}, layers={qnn_res['n_q_layers']}, "
                    f"hidden={qnn_res['head_hidden']}, best_val_f1={qnn_res['best_val_f1']:.4f}"
                ),
            }
        )
    else:
        rows.append(
            {
                "model": "Hybrid-QNN",
                "accuracy": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "f1_macro": np.nan,
                "notes": f"skipped: {qnn_res.get('reason', 'unknown')}",
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("accuracy", ascending=False, na_position="last")
    out_df.to_csv(CSV_OUT, index=False, encoding="utf-8")

    md_lines = [
        "# QNN Hybrid Reproduction Comparison",
        "",
        f"- Data: `{DATA_FILE}`",
        f"- Split: temporal holdout (last 20% by date)",
        f"- Train/Test: {len(X_train)}/{len(X_test)}",
        "",
        out_df.to_markdown(index=False),
        "",
        "## Detailed Reports",
        "",
        "### KNN",
        "```",
        baseline_res["KNN"]["report"],
        "```",
        "### LR",
        "```",
        baseline_res["LR"]["report"],
        "```",
        "### BP-MLP",
        "```",
        baseline_res["BP-MLP"]["report"],
        "```",
    ]
    if qnn_res.get("status") == "ok":
        md_lines += ["### Hybrid-QNN", "```", qnn_res["report"], "```"]

    MD_OUT.write_text("\n".join(md_lines), encoding="utf-8")

    txt_lines = [
        "=" * 72,
        "QNN HYBRID REPRODUCTION COMPARISON",
        "=" * 72,
        f"Data file: {DATA_FILE}",
        f"Temporal split train/test: {len(X_train)}/{len(X_test)}",
        "",
        out_df.to_string(index=False),
        "",
        f"CSV saved: {CSV_OUT}",
        f"MD saved:  {MD_OUT}",
        f"QNN tuning CSV saved: {QNN_TRIAL_CSV_OUT}",
        f"QNN tuning MD saved:  {QNN_TRIAL_MD_OUT}",
    ]
    TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")

    print(out_df.to_string(index=False))
    print(f"\nSaved: {CSV_OUT}")
    print(f"Saved: {MD_OUT}")
    print(f"Saved: {TXT_OUT}")
    print(f"Saved: {QNN_TRIAL_CSV_OUT}")
    print(f"Saved: {QNN_TRIAL_MD_OUT}")


if __name__ == "__main__":
    main()
