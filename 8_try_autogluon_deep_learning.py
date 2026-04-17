"""
Step 8: AutoGluon Deep Learning AutoML Framework

AutoGluon is a state-of-the-art AutoML framework that:
- Automatically selects best models (deep learning, gradient boosting, ensembles)
- Performs hyperparameter tuning automatically
- Saves significant time vs manual model selection
- Often achieves top-tier performance without manual tuning

Outputs:
- models/autogluon_pretrain_predictor/
- models/autogluon_inplay_predictor/
- reports/autogluon_comparison.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib

# Try to import AutoGluon
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False
    print("⚠️  AutoGluon not installed. Install with: pip install autogluon")

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


def train_autogluon(
    task_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_dir: Path,
    time_limit: int = 600,  # 10 minutes per task
) -> Dict[str, Any]:
    """Train AutoGluon TabularPredictor."""
    if not HAS_AUTOGLUON:
        return {
            "success": False,
            "error": "AutoGluon not installed",
            "accuracy": 0,
            "f1_macro": 0,
            "probs": None,
        }

    print(f"\n[AutoGluon] Training {task_name} (time limit: {time_limit}s)...")
    
    # Prepare training data with labels
    train_data = X_train.copy()
    train_data["result"] = y_train
    
    test_data = X_test.copy()
    
    try:
        # Create predictor
        predictor = TabularPredictor(
            label="result",
            path=str(model_dir),
            problem_type="multiclass",
            eval_metric="f1_macro",  # Optimize for macro-F1 (fairness across classes)
            verbosity=1,
        )
        
        print(f"  Starting AutoGluon training...")
        start_time = time.time()
        
        # Fit model with early stopping and time limit
        predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            presets="best_quality",  # High-quality ensemble
            num_bag_folds=5,  # Cross-validation
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Training completed in {elapsed:.1f}s")
        
        # Get predictions
        y_pred = predictor.predict(test_data)
        y_pred_probs = predictor.predict_proba(test_data)
        
        # Evaluate
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        
        # Try to get training metrics via leaderboard
        leaderboard = predictor.leaderboard()
        best_model = leaderboard.iloc[0]
        
        # Get feature importance from best model
        feature_importance = None
        try:
            feature_importance = predictor.feature_importance(test_data)
        except:
            pass
        
        print(f"  ✓ Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        return {
            "success": True,
            "predictor": predictor,
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "probs": y_pred_probs,
            "predictions": y_pred,
            "leaderboard": leaderboard,
            "feature_importance": feature_importance,
            "elapsed": elapsed,
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "accuracy": 0,
            "f1_macro": 0,
            "probs": None,
        }


def train_one_task(
    task_name: str,
    data_file: Path,
    exclude_cols: set[str],
    model_prefix: str,
    report_file: Path,
    group_col: str | None = None,
) -> Dict:
    """Train AutoGluon on a single task."""
    print("\n" + "=" * 72)
    print(f"AutoGluon AutoML Training: {task_name}")
    print("=" * 72)

    df = pd.read_csv(data_file)
    feature_cols = _select_numeric_features(df, exclude_cols)

    X_df = df[feature_cols].fillna(0)
    y_raw = df["result"].astype(str)

    label_to_idx = {label: i for i, label in enumerate(LABEL_ORDER)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    # Map labels to numeric
    y = y_raw.map(label_to_idx)

    # Temporal split
    train_mask, test_mask = _temporal_holdout(df, date_col="date", test_frac=0.2, group_col=group_col)
    
    X_train_df = X_df[train_mask]
    y_train = y[train_mask]
    X_test_df = X_df[test_mask]
    y_test = y[test_mask]

    print(f"Split: train={len(X_train_df)}, test={len(X_test_df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Label distribution (train): {y_train.value_counts().to_dict()}")

    # Create model directory
    model_dir = MODEL_DIR / f"autogluon_{model_prefix}_predictor"
    
    # Train AutoGluon
    result = train_autogluon(
        task_name=task_name,
        X_train=X_train_df.reset_index(drop=True),
        X_test=X_test_df.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        model_dir=model_dir,
        time_limit=600,  # 10 minutes per task
    )

    # Per-minute analysis if in-play
    minute_text = ""
    if task_name.lower().startswith("in-play") and "minute" in df.columns and result["success"]:
        minute_text = "\n=== Performance by Minute ===\n"
        test_df = df.loc[test_mask].copy()
        test_df["pred"] = [idx_to_label[int(p)] for p in result["predictions"]]
        
        minute_acc = {}
        for minute, group in test_df.groupby("minute"):
            acc = accuracy_score(group["result"], group["pred"])
            minute_acc[minute] = acc
        
        for minute in sorted(minute_acc.keys()):
            minute_text += f"  {int(minute):2d}': {minute_acc[minute]:.4f}\n"

    return {
        "task_name": task_name,
        "success": result["success"],
        "result": result,
        "feature_count": len(feature_cols),
        "test_size": len(X_test_df),
        "minute_text": minute_text,
    }


def main() -> None:
    print("\n" + "=" * 72)
    print("AUTOGLUON DEEP LEARNING AUTOML")
    print("=" * 72)

    if not HAS_AUTOGLUON:
        print("\n⚠️  AutoGluon is not installed!")
        print("Install with: pip install autogluon")
        return

    # Pre-match task
    print("\n[1/2] Starting Pre-match prediction task...")
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
        model_prefix="pretrain",
        report_file=REPORT_DIR / "autogluon_comparison.txt",
        group_col=None,
    )

    # In-play task
    print("\n[2/2] Starting In-play prediction task...")
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
        model_prefix="inplay",
        report_file=REPORT_DIR / "autogluon_comparison.txt",
        group_col="fixture_id",
    )

    # Generate comparison report
    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("AUTOGLUON AUTOML COMPARISON REPORT")
    report_lines.append("=" * 72)
    report_lines.append("")

    for task_result in [pretrain_result, inplay_result]:
        report_lines.append(f"\n### {task_result['task_name']} ###")
        report_lines.append(f"Features: {task_result['feature_count']}, Test samples: {task_result['test_size']}")
        report_lines.append("")
        
        if task_result["success"]:
            result = task_result["result"]
            report_lines.append(f"Status: ✓ SUCCESS")
            report_lines.append(f"Accuracy: {result['accuracy']:.4f}")
            report_lines.append(f"Macro-F1: {result['f1_macro']:.4f}")
            report_lines.append(f"Training time: {result['elapsed']:.1f}s")
            report_lines.append("")
            
            # Show leaderboard
            if result.get("leaderboard") is not None:
                report_lines.append("Top Models (Leaderboard):")
                leaderboard = result["leaderboard"].head(10)
                for idx, row in leaderboard.iterrows():
                    model_name = row.get("model", "Unknown")
                    score = row.get("score_test", 0)
                    report_lines.append(f"  {model_name}: {score:.4f}")
                report_lines.append("")
            
            # Feature importance
            if result.get("feature_importance") is not None:
                report_lines.append("Top 10 Important Features:")
                fi = result["feature_importance"].head(10)
                for idx, (feat, imp) in enumerate(fi.items(), 1):
                    report_lines.append(f"  {idx}. {feat}: {imp:.4f}")
                report_lines.append("")
        else:
            result = task_result["result"]
            report_lines.append(f"Status: ✗ FAILED")
            report_lines.append(f"Error: {result.get('error', 'Unknown')}")
            report_lines.append("")
        
        if task_result["minute_text"]:
            report_lines.append(task_result["minute_text"])

    report_text = "\n".join(report_lines)
    report_path = REPORT_DIR / "autogluon_comparison.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + "=" * 72)
    print("[OK] AutoGluon AutoML pipeline complete!")
    print(f"Report saved: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
