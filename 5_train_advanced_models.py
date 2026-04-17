"""
Step 5: Advanced model training (new experimental pipeline)

What is different from baseline:
1) Temporal holdout split by date (more realistic than random split)
2) Fixture-level holdout for in-play data (prevents leakage across minutes)
3) Stronger ensemble candidates (RF / ExtraTrees / HistGB / Stacking)
4) Multi-metric model selection (primary: macro F1, secondary: accuracy)

Outputs:
  - models/pretrain_model_advanced.pkl
  - models/inplay_model_advanced.pkl
  - reports/pretrain_report_advanced.txt
  - reports/inplay_report_advanced.txt
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, roc_auc_score

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _select_numeric_features(df: pd.DataFrame, exclude_cols: set[str]) -> List[str]:
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    return [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

# The following functions are defined in the advanced training script for better modularity and clarity.
def _temporal_holdout(
    df: pd.DataFrame,
    date_col: str,
    test_frac: float = 0.2,
    group_col: str | None = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns train_mask, test_mask.

    If group_col is provided, split by unique groups ordered by first date.
    This is critical for in-play data, where one fixture has multiple minute rows.
    """
    tmp = df.copy()
    tmp[date_col] = _to_datetime(tmp[date_col])
    # If no group_col, simply split by date
    if group_col is None:
        tmp = tmp.sort_values(date_col)
        n_test = max(1, int(len(tmp) * test_frac))
        test_idx = tmp.tail(n_test).index
        test_mask = df.index.isin(test_idx)
        train_mask = ~test_mask
        return pd.Series(train_mask, index=df.index), pd.Series(test_mask, index=df.index)
    # If group_col is provided, split by unique groups ordered by first date
    group_df = (
        tmp[[group_col, date_col]]
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[group_col], keep="first")
    )
    n_test_groups = max(1, int(len(group_df) * test_frac))
    test_groups = set(group_df.tail(n_test_groups)[group_col].tolist())

    test_mask = df[group_col].isin(test_groups)
    train_mask = ~test_mask
    return train_mask, test_mask


def _build_candidates(random_state: int = 42, include_stacking: bool = True) -> Dict[str, object]:
    # Candidate 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=450,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    # Candidate 2: Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_split=6,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    # Candidate 3: Histogram Gradient Boosting
    hgb = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.05,
        max_iter=350,
        l2_regularization=1.0,
        random_state=random_state,
    )

    candidates = {
        "RandomForest": rf,
        "ExtraTrees": et,
        "HistGradientBoosting": hgb,
    }
    # Candidate 4: Stacking ensemble of the above (optional, can be time-consuming to train)
    if include_stacking:
        # Candidate 4: Stacking ensemble
        stack = StackingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=260,
                        max_depth=10,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=260,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.06,
                        max_iter=250,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            stack_method="predict_proba",
            passthrough=True,
            n_jobs=-1,
            cv=2,
        )
        candidates["Stacking"] = stack

    return candidates


def _evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            metrics["log_loss"] = float(log_loss(y_test, y_prob, labels=model.classes_))
        except Exception:
            metrics["log_loss"] = float("nan")
        try:
            from sklearn.preprocessing import label_binarize

            y_test_bin = label_binarize(y_test, classes=model.classes_)
            metrics["roc_auc_weighted_ovr"] = float(
                roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")
            )
        except Exception:
            metrics["roc_auc_weighted_ovr"] = float("nan")
    else:
        metrics["log_loss"] = float("nan")
        metrics["roc_auc_weighted_ovr"] = float("nan")

    return metrics


def _format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"F1-macro={metrics['f1_macro']:.4f}, "
        f"LogLoss={metrics['log_loss']:.4f}, "
        f"ROC-AUC(w-ovr)={metrics['roc_auc_weighted_ovr']:.4f}"
    )


def train_advanced_model(
    task_name: str,
    data_file: Path,
    exclude_cols: set[str],
    model_file: Path,
    report_file: Path,
    group_col: str | None = None,
    include_stacking: bool = True,
) -> None:
    print("\n" + "=" * 72)
    print(f"Advanced training for: {task_name}")
    print("=" * 72)

    df = pd.read_csv(data_file)
    print(f"Dataset shape: {df.shape}")

    feature_cols = _select_numeric_features(df, exclude_cols)
    X = df[feature_cols].copy().fillna(0)
    y = df["result"].copy()

    train_mask, test_mask = _temporal_holdout(df, date_col="date", test_frac=0.2, group_col=group_col)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    print("Test period label distribution:")
    print(y_test.value_counts())

    candidates = _build_candidates(random_state=42, include_stacking=include_stacking)

    best_name = None
    best_model = None
    best_metrics = None

    for name, model in candidates.items():
        print("\n" + "-" * 72)
        print(f"Training candidate: {name}")
        model.fit(X_train, y_train)

        metrics = _evaluate_model(model, X_test, y_test)
        print(_format_metrics(metrics))

        if best_metrics is None:
            best_name, best_model, best_metrics = name, model, metrics
            continue

        # Primary criterion: higher macro F1; secondary: higher accuracy
        if (
            metrics["f1_macro"] > best_metrics["f1_macro"]
            or (
                np.isclose(metrics["f1_macro"], best_metrics["f1_macro"])
                and metrics["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_name, best_model, best_metrics = name, model, metrics

    print("\n" + "=" * 72)
    print(f"Best model for {task_name}: {best_name}")
    print(_format_metrics(best_metrics))

    y_pred_best = best_model.predict(X_test)
    cls_report = classification_report(y_test, y_pred_best, target_names=["Away Win", "Draw", "Home Win"])
    cm = confusion_matrix(y_test, y_pred_best)

    minute_accuracy_text = ""
    if task_name.lower().startswith("in-play") and "minute" in df.columns:
        eval_df = df.loc[test_mask].copy()
        eval_df["pred"] = y_pred_best
        minute_acc = eval_df.groupby("minute").apply(lambda t: accuracy_score(t["result"], t["pred"]))
        minute_accuracy_text = "\nAccuracy by minute (test holdout):\n"
        for minute, acc in minute_acc.items():
            minute_accuracy_text += f"  {int(minute):2d}': {acc:.4f}\n"

    payload = {
        "model": best_model,
        "model_name": best_name,
        "features": feature_cols,
        "scaler": None,
        "metrics": best_metrics,
        "split_strategy": "temporal last 20% by date",
        "group_holdout": group_col,
    }

    with open(model_file, "wb") as f:
        pickle.dump(payload, f)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"ADVANCED REPORT - {task_name}\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write("Split strategy: temporal last 20% by date\n")
        if group_col:
            f.write(f"Group holdout column: {group_col}\n")
        f.write(f"Feature count: {len(feature_cols)}\n\n")

        f.write("Best model metrics:\n")
        f.write(_format_metrics(best_metrics) + "\n\n")

        f.write("Classification report:\n")
        f.write(cls_report + "\n")

        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n")

        if minute_accuracy_text:
            f.write("\n" + minute_accuracy_text)

        if hasattr(best_model, "feature_importances_"):
            fi = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            f.write("\nTop 20 features:\n")
            f.write(fi.head(20).to_string(index=False) + "\n")

    print(f"Model saved:  {model_file}")
    print(f"Report saved: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train advanced football prediction models")
    parser.add_argument(
        "--task",
        choices=["all", "pre", "inplay"],
        default="all",
        help="Which task to train",
    )
    args = parser.parse_args()

    if args.task in ("all", "pre"):
        train_advanced_model(
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
        model_file=MODEL_DIR / "pretrain_model_advanced.pkl",
        report_file=REPORT_DIR / "pretrain_report_advanced.txt",
        group_col=None,
        include_stacking=True,
    )

    if args.task in ("all", "inplay"):
        train_advanced_model(
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
        model_file=MODEL_DIR / "inplay_model_advanced.pkl",
        report_file=REPORT_DIR / "inplay_report_advanced.txt",
        group_col="fixture_id",
        include_stacking=False,
    )

    print("\n" + "=" * 72)
    print("[OK] Advanced training pipeline complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()
