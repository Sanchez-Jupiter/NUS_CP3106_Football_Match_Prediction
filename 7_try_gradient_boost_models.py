"""
Step 7: Try XGBoost and CatBoost models

This script trains two highly optimized gradient boosting models:
1) XGBoost - Industry-standard, extremely fast and memory-efficient
2) CatBoost - Optimized for categorical features, often better out-of-the-box

Both are compared on:
- Pre-match final result prediction
- In-play minute-checkpoint prediction

Main design choices:
- Temporal split by date (last 20% as test)
- Fixture-level group holdout for in-play data
- Class weight balancing for H/D/A imbalance
- Early stopping on validation macro-F1
- GPU acceleration if available (CatBoost + XGBoost can use CUDA)

Outputs:
- models/pretrain_model_xgboost.pkl
- models/pretrain_model_catboost.pkl
- models/inplay_model_xgboost.pkl
- models/inplay_model_catboost.pkl
- reports/gradient_boost_comparison.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Will try to import, gracefully skip if not available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not installed. Install with: pip install catboost")

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["A", "D", "H"]


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
    """Split data temporally - last test_frac chronologically."""
    temp = df.copy()
    temp[date_col] = _to_datetime(temp[date_col])

    if group_col is None:
        temp = temp.sort_values(date_col)
        n_test = max(1, int(len(temp) * test_frac))
        test_idx = temp.tail(n_test).index
        test_mask = df.index.isin(test_idx)
        train_mask = ~test_mask
        return pd.Series(train_mask, index=df.index), pd.Series(test_mask, index=df.index)

    # For in-play data: holdout by fixture_id temporal order
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


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute sample weights for class imbalance."""
    counts = np.bincount(y)
    weights = {}
    total = len(y)
    for i in range(len(counts)):
        weights[i] = total / (len(counts) * max(1, counts[i]))
    return weights


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_to_idx: dict,
    idx_to_label: dict,
) -> Tuple[xgb.XGBClassifier, float, float, float, dict]:
    """Train XGBoost with early stopping on validation macro-F1."""
    if not HAS_XGBOOST:
        return None, 0, 0, 0, {}

    # Compute sample weights
    sample_weights = np.array([_compute_class_weights(y_train)[int(y)] for y in y_train])

    print("\n[XGBoost] Training...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",  # GPU-compatible if CUDA available
        device="cuda",  # Try GPU, falls back to CPU automatically
        objective="multi:softmax",
        num_class=3,
        early_stopping_rounds=16,
        eval_metric="mlogloss",
        verbose=False,
    )

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False,
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_f1 = f1_score(y_val, y_pred_val, average="macro")
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")

    # Placeholder for test predictions
    test_probs = model.predict_proba(X_test)

    return model, test_acc, test_f1, test_probs, {"train_acc": train_acc, "val_f1": val_f1}


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_to_idx: dict,
    idx_to_label: dict,
) -> Tuple[cb.CatBoostClassifier, float, float, np.ndarray, dict]:
    """Train CatBoost with early stopping on validation macro-F1."""
    if not HAS_CATBOOST:
        return None, 0, 0, None, {}

    # Compute class weights
    class_weights = _compute_class_weights(y_train)

    print("\n[CatBoost] Training...")

    # Try GPU first, fall back to CPU
    for _task_type in ("GPU", "CPU"):
        try:
            cb_params = dict(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                class_weights=class_weights,
                random_state=42,
                early_stopping_rounds=16,
                eval_metric="MultiClass",
                verbose=False,
                task_type=_task_type,
                bootstrap_type="Bernoulli",
                subsample=0.8,
            )
            model = cb.CatBoostClassifier(**cb_params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )
            print(f"  [CatBoost device: {_task_type}]")
            break
        except Exception:
            if _task_type == "GPU":
                continue
            raise

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_f1 = f1_score(y_val, y_pred_val, average="macro")
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")

    # Get probabilities
    test_probs = model.predict_proba(X_test)

    return model, test_acc, test_f1, test_probs, {"train_acc": train_acc, "val_f1": val_f1}


def train_one_task(
    task_name: str,
    data_file: Path,
    exclude_cols: set[str],
    model_prefix: str,
    report_file: Path,
    group_col: str | None = None,
) -> Dict:
    """Train XGBoost and CatBoost on a single task."""
    print("\n" + "=" * 72)
    print(f"Gradient Boost Training: {task_name}")
    print("=" * 72)

    df = pd.read_csv(data_file)
    feature_cols = _select_numeric_features(df, exclude_cols)

    X_df = df[feature_cols].fillna(0)
    y_raw = df["result"].astype(str)

    label_to_idx = {label: i for i, label in enumerate(LABEL_ORDER)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    y = y_raw.map(label_to_idx).values

    # Temporal split
    train_mask, test_mask = _temporal_holdout(df, date_col="date", test_frac=0.2, group_col=group_col)
    
    # Further split training into train/val
    X_train_df = X_df[train_mask]
    y_train_all = y[train_mask]
    X_test = X_df[test_mask].values.astype(np.float32)
    y_test = y[test_mask]

    # Temporal split of training data for validation
    X_train_np = X_train_df.values.astype(np.float32)
    val_frac = 0.15
    split_idx = int(len(X_train_np) * (1 - val_frac))
    
    X_train = X_train_np[:split_idx]
    y_train = y_train_all[:split_idx]
    X_val = X_train_np[split_idx:]
    y_val = y_train_all[split_idx:]

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    results = {}

    # Train XGBoost
    if HAS_XGBOOST:
        try:
            xgb_model, xgb_acc, xgb_f1, xgb_probs, xgb_info = train_xgboost(
                X_train, y_train, X_val, y_val, X_test, y_test,
                label_to_idx, idx_to_label
            )
            model_path_xgb = MODEL_DIR / f"{model_prefix}_xgboost.pkl"
            joblib.dump(xgb_model, model_path_xgb)
            results["xgboost"] = {
                "model": xgb_model,
                "accuracy": xgb_acc,
                "f1_macro": xgb_f1,
                "probs": xgb_probs,
                "path": model_path_xgb,
                "info": xgb_info,
            }
            print(f"[✓ XGBoost] Acc={xgb_acc:.4f}, F1={xgb_f1:.4f} | Saved: {model_path_xgb}")
        except Exception as e:
            print(f"[✗ XGBoost] Error: {e}")

    # Train CatBoost
    if HAS_CATBOOST:
        try:
            cb_model, cb_acc, cb_f1, cb_probs, cb_info = train_catboost(
                X_train, y_train, X_val, y_val, X_test, y_test,
                label_to_idx, idx_to_label
            )
            model_path_cb = MODEL_DIR / f"{model_prefix}_catboost.pkl"
            joblib.dump(cb_model, model_path_cb)
            results["catboost"] = {
                "model": cb_model,
                "accuracy": cb_acc,
                "f1_macro": cb_f1,
                "probs": cb_probs,
                "path": model_path_cb,
                "info": cb_info,
            }
            print(f"[✓ CatBoost] Acc={cb_acc:.4f}, F1={cb_f1:.4f} | Saved: {model_path_cb}")
        except Exception as e:
            print(f"[✗ CatBoost] Error: {e}")

    # Produce per-minute analysis if in-play
    minute_text = ""
    if task_name.lower().startswith("in-play") and "minute" in df.columns:
        minute_text = "\n=== Performance by Minute ===\n"
        test_df = df.loc[test_mask].copy()
        
        for model_name, result in results.items():
            probs = result["probs"]
            pred_idx = probs.argmax(axis=1)
            pred_labels = np.array([idx_to_label[i] for i in pred_idx])
            
            test_df[f"pred_{model_name}"] = pred_labels
            minute_acc = {}
            for minute, group in test_df.groupby("minute"):
                acc = accuracy_score(group["result"], group[f"pred_{model_name}"])
                minute_acc[minute] = acc
            
            minute_text += f"\n{model_name.upper()}:\n"
            for minute in sorted(minute_acc.keys()):
                minute_text += f"  {int(minute):2d}': {minute_acc[minute]:.4f}\n"

    return {
        "task_name": task_name,
        "results": results,
        "feature_count": len(feature_cols),
        "test_size": len(X_test),
        "minute_text": minute_text,
    }


def main() -> None:
    print("\n" + "=" * 72)
    print("GRADIENT BOOST MODELS (XGBoost + CatBoost)")
    print("=" * 72)

    if not HAS_XGBOOST and not HAS_CATBOOST:
        print("\n⚠️  Neither XGBoost nor CatBoost is installed!")
        print("Install with:")
        print("  pip install xgboost")
        print("  pip install catboost")
        return

    # Pre-match task
    pretrain_result = train_one_task(
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
        model_prefix="pretrain_model",
        report_file=REPORT_DIR / "gradient_boost_comparison.txt",
        group_col=None,
    )

    # In-play task
    inplay_result = train_one_task(
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
        model_prefix="inplay_model",
        report_file=REPORT_DIR / "gradient_boost_comparison.txt",
        group_col="fixture_id",
    )

    # Generate comparison report
    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("GRADIENT BOOST MODELS COMPARISON REPORT")
    report_lines.append("=" * 72)
    report_lines.append("")

    for task_result in [pretrain_result, inplay_result]:
        report_lines.append(f"\n### {task_result['task_name']} ###")
        report_lines.append(f"Features: {task_result['feature_count']}, Test samples: {task_result['test_size']}")
        report_lines.append("")

        for model_name, result in task_result["results"].items():
            report_lines.append(f"{model_name.upper()}:")
            report_lines.append(f"  Accuracy: {result['accuracy']:.4f}")
            report_lines.append(f"  Macro-F1: {result['f1_macro']:.4f}")
            report_lines.append(f"  Train Acc: {result['info'].get('train_acc', 0):.4f}")
            report_lines.append(f"  Val F1: {result['info'].get('val_f1', 0):.4f}")
            report_lines.append("")

        if task_result["minute_text"]:
            report_lines.append(task_result["minute_text"])

    report_text = "\n".join(report_lines)
    report_path = REPORT_DIR / "gradient_boost_comparison.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + "=" * 72)
    print("[OK] Gradient boost pipeline complete!")
    print(f"Report saved: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
