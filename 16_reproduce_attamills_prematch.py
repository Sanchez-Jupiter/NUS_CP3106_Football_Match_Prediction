"""
Strict pre-match adaptation of the 2024 Atta Mills et al. paper.

Paper:
Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques

This script intentionally does not use half-time goals, half-time results, or betting odds,
because those are not available in this project for a strict pre-match setting.

What this script does:
1. Uses the real engineered pre-match dataset in this repo.
2. Trains paper-style tabular models for three-class result prediction.
3. Evaluates with a strict temporal fixture holdout instead of random K-fold.
4. Exports a report-ready summary and detailed classification reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import RandomOverSampler

    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


warnings.filterwarnings("ignore")

DATA_FILE = Path("data/processed/pretrain_dataset.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = REPORT_DIR / "16_attamills_prematch_results.csv"
TXT_OUT = REPORT_DIR / "16_attamills_prematch_report.txt"
JSON_OUT = REPORT_DIR / "16_attamills_prematch_report.json"

SEED = 42
LABEL_ORDER = ["A", "D", "H"]

CATEGORICAL_COLS = [
    "home_team",
    "away_team",
    "home_formation",
    "away_formation",
]

NUMERIC_COLS = [
    "h_games_played",
    "h_recent_wins",
    "h_recent_draws",
    "h_recent_losses",
    "h_recent_gf",
    "h_recent_ga",
    "h_recent_gd",
    "h_win_rate",
    "h_avg_gf",
    "h_avg_ga",
    "a_games_played",
    "a_recent_wins",
    "a_recent_draws",
    "a_recent_losses",
    "a_recent_gf",
    "a_recent_ga",
    "a_recent_gd",
    "a_win_rate",
    "a_avg_gf",
    "a_avg_ga",
    "h_points_per_game",
    "a_points_per_game",
    "h_clean_sheet_rate",
    "a_clean_sheet_rate",
    "h_failed_to_score_rate",
    "a_failed_to_score_rate",
    "h_avg_first_half_gf",
    "a_avg_first_half_gf",
    "h_avg_second_half_gf",
    "a_avg_second_half_gf",
    "h_avg_yellow",
    "a_avg_yellow",
    "h_avg_red",
    "a_avg_red",
    "h2h_games",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_home_goal_diff_avg",
    "h_days_since_last_match",
    "a_days_since_last_match",
    "h_matches_last_7d",
    "a_matches_last_7d",
    "h_matches_last_14d",
    "a_matches_last_14d",
    "round_no",
    "season_progress",
    "h_rank",
    "a_rank",
    "h_gap_top",
    "a_gap_top",
    "h_gap_top4",
    "a_gap_top4",
    "h_gap_safety",
    "a_gap_safety",
    "importance_sum",
    "importance_diff",
    "h_key_players_started",
    "a_key_players_started",
    "h_key_players_absent",
    "a_key_players_absent",
    "h_key_players_form_avg_rating",
    "a_key_players_form_avg_rating",
    "h_key_players_form_avg_contrib",
    "a_key_players_form_avg_contrib",
    "h_starting11_avg_minutes_7d",
    "a_starting11_avg_minutes_7d",
    "h_starting11_avg_matches_7d",
    "a_starting11_avg_matches_7d",
    "diff_win_rate",
    "diff_avg_gf",
    "diff_avg_ga",
    "diff_points_per_game",
    "diff_second_half_gf",
    "diff_days_since_last_match",
    "diff_matches_last_7d",
    "diff_key_players_absent",
]


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def temporal_holdout_by_fixture(
    df: pd.DataFrame,
    date_col: str = "date",
    group_col: str = "fixture_id",
    test_frac: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    fixture_dates = (
        df[[group_col, date_col]]
        .assign(**{date_col: _to_datetime(df[date_col])})
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[group_col], keep="first")
    )
    n_test = max(1, int(len(fixture_dates) * test_frac))
    test_ids = set(fixture_dates.tail(n_test)[group_col].tolist())
    test_mask = df[group_col].isin(test_ids)
    return ~test_mask, test_mask


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df["date"] = _to_datetime(df["date"])
    df = df.dropna(subset=["date", "result"]).sort_values("date").reset_index(drop=True)

    needed = CATEGORICAL_COLS + NUMERIC_COLS + ["fixture_id", "date", "result"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required pre-match columns: {missing}")

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    categorical_pipe = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    from sklearn.pipeline import Pipeline

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_pipe), NUMERIC_COLS),
            ("cat", Pipeline(categorical_pipe), CATEGORICAL_COLS),
        ]
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def maybe_resample(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if not HAS_IMBLEARN:
        return X_train, y_train, "none"

    ros = RandomOverSampler(random_state=SEED)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    return X_res, y_res, "RandomOverSampler"


def make_models(label_encoder: LabelEncoder) -> Dict[str, object]:
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=SEED,
        ),
        "NaiveBayes": GaussianNB(),
        "FNN": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=1,
        )

    return models


def soft_vote(probability_maps: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(list(probability_maps.values()), axis=0)
    avg_prob = stacked.mean(axis=0)
    pred_idx = np.argmax(avg_prob, axis=1)
    pred = np.asarray([LABEL_ORDER[idx] for idx in pred_idx], dtype=object)
    return pred, avg_prob


def main() -> None:
    print("\n" + "=" * 72)
    print("Strict pre-match Atta Mills reproduction")
    print("=" * 72)

    df = load_dataset()
    train_mask, test_mask = temporal_holdout_by_fixture(df)
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    print(f"Train fixtures: {train_df['fixture_id'].nunique()} | rows: {len(train_df)}")
    print(f"Test fixtures:  {test_df['fixture_id'].nunique()} | rows: {len(test_df)}")

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[CATEGORICAL_COLS + NUMERIC_COLS])
    X_test = preprocessor.transform(test_df[CATEGORICAL_COLS + NUMERIC_COLS])

    label_encoder = LabelEncoder()
    y_train_idx = label_encoder.fit_transform(train_df["result"].astype(str))
    y_test_idx = label_encoder.transform(test_df["result"].astype(str))
    y_test = label_encoder.inverse_transform(y_test_idx)

    X_train_res, y_train_res, sampler_name = maybe_resample(X_train, y_train_idx)
    print(f"Sampler: {sampler_name}")

    models = make_models(label_encoder)
    results = []
    probability_maps: Dict[str, np.ndarray] = {}
    detailed_reports: Dict[str, str] = {}
    confusion_maps: Dict[str, list[list[int]]] = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_res, y_train_res)
        pred_idx = model.predict(X_test)
        pred = label_encoder.inverse_transform(pred_idx)

        metrics = metric_dict(y_test, pred)
        report = classification_report(y_test, pred, labels=LABEL_ORDER, digits=4, zero_division=0)
        confusion = confusion_matrix(y_test, pred, labels=LABEL_ORDER)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            probability_maps[name] = proba

        detailed_reports[name] = report
        confusion_maps[name] = confusion.tolist()
        results.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        })

        print(f"{name}: accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

    if len(probability_maps) >= 2:
        vote_pred, _ = soft_vote(probability_maps)
        vote_metrics = metric_dict(y_test, vote_pred)
        vote_report = classification_report(y_test, vote_pred, labels=LABEL_ORDER, digits=4, zero_division=0)
        vote_confusion = confusion_matrix(y_test, vote_pred, labels=LABEL_ORDER)

        detailed_reports["Voting"] = vote_report
        confusion_maps["Voting"] = vote_confusion.tolist()
        results.append({
            "model": "Voting",
            "accuracy": vote_metrics["accuracy"],
            "precision_macro": vote_metrics["precision_macro"],
            "recall_macro": vote_metrics["recall_macro"],
            "f1_macro": vote_metrics["f1_macro"],
        })
        print(f"Voting: accuracy={vote_metrics['accuracy']:.4f}, f1_macro={vote_metrics['f1_macro']:.4f}")

    result_df = pd.DataFrame(results).sort_values(["accuracy", "f1_macro"], ascending=False).reset_index(drop=True)
    result_df.to_csv(CSV_OUT, index=False)

    lines = [
        "Strict pre-match Atta Mills reproduction",
        "=" * 72,
        f"Data file: {DATA_FILE}",
        f"Train fixtures: {train_df['fixture_id'].nunique()} | rows: {len(train_df)}",
        f"Test fixtures: {test_df['fixture_id'].nunique()} | rows: {len(test_df)}",
        f"Sampler: {sampler_name}",
        "",
        result_df.to_string(index=False),
        "",
    ]

    for name in result_df["model"].tolist():
        lines.append("-" * 72)
        lines.append(name)
        lines.append("-" * 72)
        lines.append(detailed_reports[name])
        lines.append("Confusion matrix [A, D, H]:")
        lines.append(str(np.asarray(confusion_maps[name])))
        lines.append("")

    TXT_OUT.write_text("\n".join(lines), encoding="utf-8")
    JSON_OUT.write_text(
        json.dumps(
            {
                "results": results,
                "reports": detailed_reports,
                "confusion_matrices": confusion_maps,
                "sampler": sampler_name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"- {CSV_OUT}")
    print(f"- {TXT_OUT}")
    print(f"- {JSON_OUT}")


if __name__ == "__main__":
    main()
