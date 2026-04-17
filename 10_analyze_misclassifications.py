"""
Step 10: Analyze common misclassification patterns

This script loads the trained pre-match and in-play models, reconstructs the
test split used during training, and analyzes where the models are most often
wrong. The goal is not only to count errors, but also to highlight likely
feature patterns behind the most common confusions.

Outputs:
- reports/misclassification_analysis.txt
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["A", "D", "H"]
LABEL_DISPLAY = {"A": "Away Win", "D": "Draw", "H": "Home Win"}


def _format_label(label: str) -> str:
    return LABEL_DISPLAY.get(label, str(label))


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _apply_scaler_if_needed(bundle: dict, X: pd.DataFrame) -> np.ndarray | pd.DataFrame:
    scaler = bundle.get("scaler")
    if scaler is None:
        return X
    return scaler.transform(X)


def _predict_with_bundle_model(bundle: dict, X_input: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = bundle["model"]
    model_name = str(bundle.get("model_name", ""))

    if model_name == "XGBoost" or model.__class__.__module__.startswith("xgboost"):
        model.set_params(device="cpu")
        booster = model.get_booster()
        booster.set_param({"device": "cpu"})

    pred = model.predict(X_input)
    prob = model.predict_proba(X_input)
    return np.asarray(pred), np.asarray(prob)


def _prepare_pretrain_test(bundle: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "pretrain_dataset.csv")
    features = bundle["features"]
    X = df[features].copy().fillna(0)

    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        y_raw = df["result"].astype(str).values
        y = label_encoder.transform(y_raw)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            df.index.values,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        X_test_input = _apply_scaler_if_needed(bundle, X_test)
        pred, prob = _predict_with_bundle_model(bundle, X_test_input)
        y_test_labels = label_encoder.inverse_transform(np.asarray(y_test).astype(int))
        pred_labels = label_encoder.inverse_transform(np.asarray(pred).astype(int))
    else:
        y = df["result"].astype(str).values
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            df.index.values,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        X_test_input = _apply_scaler_if_needed(bundle, X_test)
        pred, prob = _predict_with_bundle_model(bundle, X_test_input)
        y_test_labels = np.asarray(y_test).astype(str)
        pred_labels = np.asarray(pred).astype(str)

    eval_df = df.loc[idx_test].copy().reset_index(drop=True)
    return eval_df, y_test_labels, pred_labels, np.asarray(prob)


def _prepare_inplay_test(bundle: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "inplay_dataset.csv")
    features = bundle["features"]
    X = df[features].copy().fillna(0)
    y = df["result"].astype(str).values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index.values,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_test_input = _apply_scaler_if_needed(bundle, X_test)
    pred, prob = _predict_with_bundle_model(bundle, X_test_input)
    eval_df = df.loc[idx_test].copy().reset_index(drop=True)
    return eval_df, np.asarray(y_test).astype(str), np.asarray(pred).astype(str), np.asarray(prob)


def _top_confusions(y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 5) -> list[tuple[str, str, int]]:
    pairs: dict[tuple[str, str], int] = {}
    for actual, pred in zip(y_true, y_pred):
        if actual == pred:
            continue
        pairs[(str(actual), str(pred))] = pairs.get((str(actual), str(pred)), 0) + 1
    return sorted(((a, p, c) for (a, p), c in pairs.items()), key=lambda item: item[2], reverse=True)[:top_n]


def _feature_gap_report(
    df: pd.DataFrame,
    actual: str,
    pred: str,
    feature_cols: Iterable[str],
    top_n: int = 5,
) -> list[str]:
    wrong_mask = (df["actual"] == actual) & (df["pred"] == pred)
    correct_mask = (df["actual"] == actual) & (df["pred"] == actual)

    wrong_df = df.loc[wrong_mask, list(feature_cols)]
    correct_df = df.loc[correct_mask, list(feature_cols)]
    if wrong_df.empty or correct_df.empty:
        return ["insufficient comparable samples"]

    wrong_mean = wrong_df.mean(axis=0)
    correct_mean = correct_df.mean(axis=0)
    pooled_std = df[list(feature_cols)].std(axis=0).replace(0, np.nan)
    gap = ((wrong_mean - correct_mean) / pooled_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    top_features = gap.abs().sort_values(ascending=False).head(top_n)

    lines = []
    for feature in top_features.index:
        direction = "higher" if gap[feature] > 0 else "lower"
        lines.append(
            f"{feature}: misclassified samples are {direction} than correctly predicted {_format_label(actual)} cases "
            f"(wrong_mean={wrong_mean[feature]:.4f}, correct_mean={correct_mean[feature]:.4f})"
        )
    return lines


def _confidence_summary(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    confidence = probs.max(axis=1)
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask
    correct_avg = float(confidence[correct_mask].mean()) if correct_mask.any() else float("nan")
    wrong_avg = float(confidence[wrong_mask].mean()) if wrong_mask.any() else float("nan")
    return f"avg confidence (correct): {correct_avg:.4f} | avg confidence (wrong): {wrong_avg:.4f}"


def _minute_error_summary(df: pd.DataFrame) -> list[str]:
    if "minute" not in df.columns:
        return []
    lines = []
    minute_stats = (
        df.assign(is_correct=df["actual"] == df["pred"])
        .groupby("minute", as_index=False)
        .agg(accuracy=("is_correct", "mean"), samples=("is_correct", "size"))
        .sort_values("minute")
    )
    for row in minute_stats.itertuples(index=False):
        lines.append(f"minute {int(row.minute):2d}: accuracy={row.accuracy:.4f}, samples={int(row.samples)}")
    return lines


def analyze_task(task_name: str, eval_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, feature_cols: list[str]) -> list[str]:
    result_lines = []
    eval_df = eval_df.copy()
    eval_df["actual"] = y_true
    eval_df["pred"] = y_pred
    eval_df["correct"] = eval_df["actual"] == eval_df["pred"]
    eval_df["confidence"] = probs.max(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    result_lines.append("=" * 72)
    result_lines.append(f"MISCLASSIFICATION ANALYSIS - {task_name.upper()}")
    result_lines.append("=" * 72)
    result_lines.append(f"overall accuracy: {acc:.4f}")
    result_lines.append(_confidence_summary(probs, y_true, y_pred))
    result_lines.append("confusion matrix [A, D, H]:")
    result_lines.append(str(cm))
    result_lines.append("")

    actual_stats = (
        eval_df.groupby("actual", as_index=False)
        .agg(samples=("actual", "size"), accuracy=("correct", "mean"))
        .sort_values("accuracy")
    )
    result_lines.append("per-class accuracy:")
    for row in actual_stats.itertuples(index=False):
        result_lines.append(f"{_format_label(row.actual)}: accuracy={row.accuracy:.4f}, samples={int(row.samples)}")
    result_lines.append("")

    confusions = _top_confusions(y_true, y_pred, top_n=5)
    result_lines.append("top confusion pairs:")
    if not confusions:
        result_lines.append("no misclassifications found")
    for actual, pred, count in confusions:
        result_lines.append(f"{_format_label(actual)} -> {_format_label(pred)}: {count} samples")
        for line in _feature_gap_report(eval_df, actual, pred, feature_cols, top_n=5):
            result_lines.append(f"  - {line}")
    result_lines.append("")

    high_conf_errors = eval_df[~eval_df["correct"]].sort_values("confidence", ascending=False).head(5)
    result_lines.append("highest-confidence errors:")
    if high_conf_errors.empty:
        result_lines.append("none")
    else:
        core_cols = [col for col in ["date", "home_team", "away_team", "minute", "actual", "pred", "confidence"] if col in high_conf_errors.columns]
        for row in high_conf_errors[core_cols].itertuples(index=False):
            row_bits = []
            for col, value in zip(core_cols, row):
                if col == "confidence":
                    row_bits.append(f"confidence={value:.4f}")
                else:
                    row_bits.append(f"{col}={value}")
            result_lines.append(" | ".join(row_bits))
    result_lines.append("")

    minute_lines = _minute_error_summary(eval_df)
    if minute_lines:
        result_lines.append("accuracy by minute:")
        result_lines.extend(minute_lines)
        result_lines.append("")

    return result_lines


def main() -> None:
    pretrain_bundle = _load_pickle(MODEL_DIR / "pretrain_model.pkl")
    inplay_bundle = _load_pickle(MODEL_DIR / "inplay_model.pkl")

    pre_df, pre_y_true, pre_y_pred, pre_probs = _prepare_pretrain_test(pretrain_bundle)
    inplay_df, inplay_y_true, inplay_y_pred, inplay_probs = _prepare_inplay_test(inplay_bundle)

    report_lines = []
    report_lines.extend(analyze_task("pre-match", pre_df, pre_y_true, pre_y_pred, pre_probs, pretrain_bundle["features"]))
    report_lines.append("")
    report_lines.extend(analyze_task("in-play", inplay_df, inplay_y_true, inplay_y_pred, inplay_probs, inplay_bundle["features"]))

    report_path = REPORT_DIR / "misclassification_analysis.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] Misclassification analysis saved to: {report_path}")


if __name__ == "__main__":
    main()