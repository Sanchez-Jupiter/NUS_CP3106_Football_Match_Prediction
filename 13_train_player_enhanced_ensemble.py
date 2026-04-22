"""
Player-enhanced pre-match ensemble inspired by the 2025 FIFA World Cup paper.

Adaptation for this project:
- strict pre-match task on data/processed/pretrain_dataset.csv
- three-class prediction: Away Win / Draw / Home Win
- temporal fixture holdout instead of random split
- player-centric features already available in this project
- dimensionality reduction + multiple classifiers + majority voting ensemble
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DATA_FILE = Path("data/processed/pretrain_dataset.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LABEL_ORDER = ["A", "D", "H"]

PLAYER_FEATURES = [
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
    "diff_key_players_absent",
]

TEAM_CONTEXT_FEATURES = [
    "home_team",
    "away_team",
    "home_formation",
    "away_formation",
    "h_win_rate",
    "a_win_rate",
    "h_avg_gf",
    "a_avg_gf",
    "h_avg_ga",
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
    "diff_win_rate",
    "diff_avg_gf",
    "diff_avg_ga",
    "diff_points_per_game",
    "diff_second_half_gf",
    "diff_days_since_last_match",
    "diff_matches_last_7d",
]


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _temporal_holdout_by_fixture(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    test_frac: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    temp = df.copy()
    temp[date_col] = _to_datetime(temp[date_col])
    group_df = (
        temp[[group_col, date_col]]
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[group_col], keep="first")
    )
    n_test_groups = max(1, int(len(group_df) * test_frac))
    test_groups = set(group_df.tail(n_test_groups)[group_col].tolist())
    test_mask = df[group_col].isin(test_groups)
    return ~test_mask, test_mask


def _train_val_split_fixture_ids(df_train: pd.DataFrame, val_frac: float = 0.15) -> tuple[set[int], set[int]]:
    fixture_df = (
        df_train[["fixture_id", "date"]]
        .assign(date=_to_datetime(df_train["date"]))
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["fixture_id"], keep="first")
    )
    split = int(len(fixture_df) * (1.0 - val_frac))
    train_ids = set(fixture_df.iloc[:split]["fixture_id"].tolist())
    val_ids = set(fixture_df.iloc[split:]["fixture_id"].tolist())
    return train_ids, val_ids


def _build_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    requested = PLAYER_FEATURES + TEAM_CONTEXT_FEATURES
    missing = [col for col in requested if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required player-enhanced features: {missing}")

    categorical_cols = [col for col in requested if not pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = [col for col in requested if col not in categorical_cols]
    feature_cols = requested
    return feature_cols, numeric_cols, categorical_cols


def _make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def _make_models() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=320,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=360,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance", metric="minkowski"),
    }


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, labels: list[str]) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
    }
    if y_prob is not None:
        try:
            from sklearn.preprocessing import label_binarize

            y_true_bin = label_binarize(y_true, classes=labels)
            metrics["roc_auc_weighted_ovr"] = float(
                roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted")
            )
        except Exception:
            metrics["roc_auc_weighted_ovr"] = float("nan")
    else:
        metrics["roc_auc_weighted_ovr"] = float("nan")
    return metrics

# Majority vote with tiebreak using average probabilities across models
def _majority_vote_with_tiebreak(predictions: list[np.ndarray], probabilities: list[np.ndarray], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_matrix = np.vstack(predictions)
    avg_prob = np.mean(np.stack(probabilities, axis=0), axis=0)
    final_pred: list[str] = []

    for idx in range(pred_matrix.shape[1]):
        votes = pred_matrix[:, idx].tolist()
        counts = Counter(votes)
        top_count = max(counts.values())
        top_labels = [label for label, count in counts.items() if count == top_count]
        if len(top_labels) == 1:
            final_pred.append(top_labels[0])
            continue

        top_label_idx = [int(np.where(labels == label)[0][0]) for label in top_labels]
        best_idx = max(top_label_idx, key=lambda label_idx: avg_prob[idx, label_idx])
        final_pred.append(str(labels[best_idx]))

    return np.asarray(final_pred, dtype=object), avg_prob


def main() -> None:
    print("\n" + "=" * 72)
    print("Player-enhanced pre-match ensemble")
    print("=" * 72)

    df = pd.read_csv(DATA_FILE)
    df["date"] = _to_datetime(df["date"])
    df = df.dropna(subset=["date", "result"]).sort_values("date").reset_index(drop=True)

    feature_cols, numeric_cols, categorical_cols = _build_feature_sets(df)

    train_mask, test_mask = _temporal_holdout_by_fixture(df, date_col="date", group_col="fixture_id", test_frac=0.2)
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)
    train_fixture_ids, val_fixture_ids = _train_val_split_fixture_ids(train_df, val_frac=0.15)

    if not val_fixture_ids:
        sorted_ids = sorted(train_df["fixture_id"].unique().tolist())
        val_fixture_ids = set(sorted_ids[-1:])
        train_fixture_ids = set(sorted_ids[:-1])

    subtrain_df = train_df[train_df["fixture_id"].isin(train_fixture_ids)].copy().reset_index(drop=True)
    val_df = train_df[train_df["fixture_id"].isin(val_fixture_ids)].copy().reset_index(drop=True)

    X_subtrain = subtrain_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_train_full = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_subtrain = subtrain_df["result"].astype(str).to_numpy()
    y_val = val_df["result"].astype(str).to_numpy()
    y_train_full = train_df["result"].astype(str).to_numpy()
    y_test = test_df["result"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ORDER)
    labels = label_encoder.classes_

    print(f"Total fixtures: {df['fixture_id'].nunique()}")
    print(f"Train/Val/Test fixtures: {len(train_fixture_ids)}/{len(val_fixture_ids)}/{test_df['fixture_id'].nunique()}")
    print(f"Train/Val/Test samples: {len(subtrain_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Feature count: {len(feature_cols)} | Numeric: {len(numeric_cols)} | Categorical: {len(categorical_cols)}")

    preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
    X_subtrain_trans = preprocessor.fit_transform(X_subtrain)
    X_val_trans = preprocessor.transform(X_val)
    X_train_full_trans = preprocessor.transform(X_train_full)
    X_test_trans = preprocessor.transform(X_test)

    svd_dim = int(min(64, max(8, X_subtrain_trans.shape[1] - 1)))
    svd = TruncatedSVD(n_components=svd_dim, random_state=SEED)
    X_subtrain_svd = svd.fit_transform(X_subtrain_trans)
    X_val_svd = svd.transform(X_val_trans)
    X_train_full_svd = svd.transform(X_train_full_trans)
    X_test_svd = svd.transform(X_test_trans)

    print(f"Post-encoding dimension: {X_subtrain_trans.shape[1]} | SVD dimension: {svd_dim}")

    candidates = _make_models()
    validation_rows: list[tuple[str, float, float]] = []

    for model_name, model in candidates.items():
        print("\n" + "-" * 72)
        print(f"Training candidate: {model_name}")
        model.fit(X_subtrain_svd, y_subtrain)
        val_pred = model.predict(X_val_svd)
        val_prob = model.predict_proba(X_val_svd) if hasattr(model, "predict_proba") else None
        metrics = _evaluate_predictions(y_val, val_pred, val_prob, LABEL_ORDER)
        validation_rows.append((model_name, metrics["f1_macro"], metrics["accuracy"]))
        print(
            f"Validation -> Accuracy={metrics['accuracy']:.4f}, "
            f"F1-macro={metrics['f1_macro']:.4f}, ROC-AUC(w-ovr)={metrics['roc_auc_weighted_ovr']:.4f}"
        )

    validation_rows.sort(key=lambda item: (item[1], item[2]), reverse=True)
    selected_names = [name for name, _, _ in validation_rows[:3]]
    print("\nSelected ensemble members:")
    for name in selected_names:
        print(f"- {name}")

    final_models: dict[str, object] = {}
    test_predictions: list[np.ndarray] = []
    test_probabilities: list[np.ndarray] = []

    for name in selected_names:
        model = _make_models()[name]
        model.fit(X_train_full_svd, y_train_full)
        final_models[name] = model
        test_predictions.append(model.predict(X_test_svd))
        test_probabilities.append(model.predict_proba(X_test_svd))

    ensemble_pred, ensemble_prob = _majority_vote_with_tiebreak(test_predictions, test_probabilities, labels)
    ensemble_metrics = _evaluate_predictions(y_test, ensemble_pred, ensemble_prob, LABEL_ORDER)
    cls_report = classification_report(y_test, ensemble_pred, labels=LABEL_ORDER, target_names=["Away Win", "Draw", "Home Win"], zero_division=0)
    cm = confusion_matrix(y_test, ensemble_pred, labels=LABEL_ORDER)

    bundle = {
        "model_type": "player_enhanced_prematch_ensemble",
        "selected_models": selected_names,
        "models": final_models,
        "preprocessor": preprocessor,
        "svd": svd,
        "features": feature_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "player_features": PLAYER_FEATURES,
        "team_context_features": TEAM_CONTEXT_FEATURES,
        "label_order": LABEL_ORDER,
        "validation_ranking": validation_rows,
        "metrics": ensemble_metrics,
    }

    model_path = MODEL_DIR / "pretrain_model_player_ensemble.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    report_lines = [
        "=" * 72,
        "PLAYER-ENHANCED PRE-MATCH ENSEMBLE REPORT",
        "=" * 72,
        "",
        f"Data file: {DATA_FILE}",
        "Paper inspiration: player-level + team-level features, dimensionality reduction, multi-model voting.",
        "Task adaptation: strict pre-match, three-class A/D/H prediction with temporal fixture holdout.",
        "",
        f"Train/Val/Test fixtures: {len(train_fixture_ids)}/{len(val_fixture_ids)}/{test_df['fixture_id'].nunique()}",
        f"Train/Val/Test samples: {len(subtrain_df)}/{len(val_df)}/{len(test_df)}",
        f"Total feature count: {len(feature_cols)}",
        f"Numeric/Categorical feature count: {len(numeric_cols)}/{len(categorical_cols)}",
        f"Post-encoding dimension: {X_subtrain_trans.shape[1]}",
        f"SVD dimension: {svd_dim}",
        "",
        "Validation ranking (by F1-macro, then accuracy):",
    ]

    for name, val_f1, val_acc in validation_rows:
        report_lines.append(f"- {name}: val_f1_macro={val_f1:.4f}, val_accuracy={val_acc:.4f}")

    report_lines += [
        "",
        f"Selected ensemble members: {', '.join(selected_names)}",
        f"Test Accuracy: {ensemble_metrics['accuracy']:.4f}",
        f"Test Macro-F1: {ensemble_metrics['f1_macro']:.4f}",
        f"Test ROC-AUC (weighted ovr): {ensemble_metrics['roc_auc_weighted_ovr']:.4f}",
        "",
        "Classification report:",
        cls_report,
        "Confusion matrix [A, D, H]:",
        str(cm),
        "",
        "Player-centric features:",
        ", ".join(PLAYER_FEATURES),
        "",
        "Team-context features:",
        ", ".join(TEAM_CONTEXT_FEATURES),
        "",
        f"Model saved to: {model_path}",
    ]

    report_path = REPORT_DIR / "pretrain_report_player_ensemble.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Final ensemble test metrics")
    print("=" * 72)
    print(f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {ensemble_metrics['f1_macro']:.4f}")
    print(f"ROC-AUC (weighted ovr): {ensemble_metrics['roc_auc_weighted_ovr']:.4f}")
    print(f"Model saved:  {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()