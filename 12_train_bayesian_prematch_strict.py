"""
Strict no-leakage Bayesian-network pre-match model.

This script is inspired by the 2017 Bayesian-network football paper, but it
uses only pre-match available information from data/processed/pretrain_dataset.csv.

Design choices:
- strict temporal holdout by fixture date (last 20% fixtures as test)
- only pre-match fields and historical/team-context statistics
- numeric feature discretization fitted on training fixtures only
- Bayesian network structure learning with pgmpy + BDeu score
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianModel

try:
    from pgmpy.estimators import BDeu
except ImportError:
    from pgmpy.estimators import BDeuScore as BDeu

from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination


DATA_FILE = Path("data/processed/pretrain_dataset.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LABEL_ORDER = ["A", "D", "H"]
TARGET_COL = "result"

STRICT_PREMATCH_FEATURES = [
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
    "h2h_games",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_home_goal_diff_avg",
    "h_days_since_last_match",
    "a_days_since_last_match",
    "h_matches_last_7d",
    "a_matches_last_7d",
    "round_no",
    "season_progress",
    "h_rank",
    "a_rank",
    "importance_sum",
    "importance_diff",
    "h_key_players_absent",
    "a_key_players_absent",
    "h_key_players_form_avg_rating",
    "a_key_players_form_avg_rating",
    "diff_win_rate",
    "diff_avg_gf",
    "diff_avg_ga",
    "diff_points_per_game",
    "diff_days_since_last_match",
    "diff_key_players_absent",
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


def _fit_feature_discretizers(train_df: pd.DataFrame, feature_cols: list[str], bins: int) -> dict[str, dict[str, Any]]:
    discretizers: dict[str, dict[str, Any]] = {}

    for col in feature_cols:
        series = train_df[col]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").fillna(0.0)
            unique_values = np.unique(values.to_numpy())
            if len(unique_values) <= bins:
                sorted_values = sorted(unique_values.tolist())
                discretizers[col] = {
                    "type": "numeric_lookup",
                    "mapping": {float(value): f"v_{idx}" for idx, value in enumerate(sorted_values)},
                    "known_values": sorted_values,
                }
            else:
                quantiles = np.linspace(0, 1, bins + 1)
                edges = np.quantile(values, quantiles)
                edges = np.unique(edges.astype(float))
                if len(edges) <= 2:
                    discretizers[col] = {"type": "constant", "value": "bin_0"}
                else:
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    discretizers[col] = {"type": "numeric_bins", "edges": edges.tolist()}
        else:
            filled = series.fillna("missing").astype(str)
            mode_value = filled.mode(dropna=False)
            fallback = str(mode_value.iloc[0]) if not mode_value.empty else "missing"
            discretizers[col] = {
                "type": "categorical",
                "known": sorted(filled.unique().tolist()),
                "fallback": fallback,
            }

    return discretizers


def _transform_with_discretizers(df: pd.DataFrame, feature_cols: list[str], discretizers: dict[str, dict[str, Any]]) -> pd.DataFrame:
    transformed = pd.DataFrame(index=df.index)

    for col in feature_cols:
        spec = discretizers[col]
        if spec["type"] == "constant":
            transformed[col] = spec["value"]
            continue

        if spec["type"] == "numeric_lookup":
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            known_values = np.asarray(spec["known_values"], dtype=float)
            mapping = {float(key): value for key, value in spec["mapping"].items()}

            def encode_value(value: float) -> str:
                value = float(value)
                if value in mapping:
                    return mapping[value]
                nearest = known_values[np.argmin(np.abs(known_values - value))]
                return mapping[float(nearest)]

            transformed[col] = values.map(encode_value).astype(str)
            continue

        if spec["type"] == "numeric_bins":
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            edges = np.asarray(spec["edges"], dtype=float)
            transformed[col] = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop").astype(str)
            continue

        if spec["type"] == "categorical":
            values = df[col].fillna(spec["fallback"]).astype(str)
            known = set(spec["known"])
            fallback = spec["fallback"]
            transformed[col] = values.map(lambda value: value if value in known else fallback)
            continue

        raise ValueError(f"Unknown discretizer type for {col}: {spec['type']}")

    transformed[TARGET_COL] = df[TARGET_COL].astype(str)
    return transformed


def _fit_bn(train_df: pd.DataFrame, feature_cols: list[str], max_indegree: int, equivalent_sample_size: int) -> BayesianModel:
    search = HillClimbSearch(train_df)
    learned = search.estimate(scoring_method=BDeu(train_df), max_indegree=max_indegree, show_progress=False)
    edges = list(learned.edges()) if hasattr(learned, "edges") else list(learned)
    model = BayesianModel(edges)

    for col in feature_cols + [TARGET_COL]:
        if col not in model.nodes():
            model.add_node(col)

    model.fit(
        train_df,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=equivalent_sample_size,
    )
    return model


def _predict_probs(model: BayesianModel, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    infer = VariableElimination(model)
    probabilities: list[np.ndarray] = []

    for _, row in test_df.iterrows():
        evidence = {col: str(row[col]) for col in feature_cols}
        query = infer.query(variables=[TARGET_COL], evidence=evidence, show_progress=False)
        state_names = [str(name) for name in query.state_names[TARGET_COL]]
        state_to_prob = {state: float(query.values[idx]) for idx, state in enumerate(state_names)}
        probabilities.append(np.array([state_to_prob.get(label, 0.0) for label in LABEL_ORDER], dtype=np.float32))

    return np.vstack(probabilities) if probabilities else np.zeros((0, len(LABEL_ORDER)), dtype=np.float32)


def _build_feature_list(df: pd.DataFrame) -> list[str]:
    missing = [col for col in STRICT_PREMATCH_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing strict pre-match features: {missing}")
    return STRICT_PREMATCH_FEATURES.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a strict no-leakage Bayesian-network pre-match model.")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of latest fixtures used for test holdout.")
    parser.add_argument("--bins", type=int, default=4, help="Quantile bins for numeric discretization.")
    parser.add_argument("--max-indegree", type=int, default=2, help="Maximum in-degree during BN structure learning.")
    parser.add_argument("--equivalent-sample-size", type=int, default=10, help="Equivalent sample size for BayesianEstimator.")
    parser.add_argument("--max-fixtures", type=int, default=0, help="Optional cap on earliest fixtures for quick experiments.")
    args = parser.parse_args()

    print("Loading strict pre-match dataset...")
    df = pd.read_csv(DATA_FILE)
    df["date"] = _to_datetime(df["date"])
    df = df.dropna(subset=["date", TARGET_COL]).sort_values("date").reset_index(drop=True)

    if args.max_fixtures > 0:
        keep_ids = df["fixture_id"].drop_duplicates().head(args.max_fixtures).tolist()
        df = df[df["fixture_id"].isin(keep_ids)].copy().reset_index(drop=True)

    feature_cols = _build_feature_list(df)
    train_mask, test_mask = _temporal_holdout_by_fixture(df, date_col="date", group_col="fixture_id", test_frac=args.test_frac)
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train or test set.")

    discretizers = _fit_feature_discretizers(train_df, feature_cols, bins=args.bins)
    train_disc = _transform_with_discretizers(train_df, feature_cols, discretizers)
    test_disc = _transform_with_discretizers(test_df, feature_cols, discretizers)

    print(f"Train fixtures: {train_df['fixture_id'].nunique()} | Test fixtures: {test_df['fixture_id'].nunique()}")
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    print(f"Feature count: {len(feature_cols)}")

    model = _fit_bn(
        train_disc,
        feature_cols,
        max_indegree=args.max_indegree,
        equivalent_sample_size=args.equivalent_sample_size,
    )

    probs = _predict_probs(model, test_disc, feature_cols)
    pred_idx = probs.argmax(axis=1)
    pred_labels = np.asarray([LABEL_ORDER[idx] for idx in pred_idx], dtype=object)
    y_true = test_disc[TARGET_COL].to_numpy(dtype=object)

    accuracy = float(accuracy_score(y_true, pred_labels))
    macro_f1 = float(f1_score(y_true, pred_labels, average="macro", labels=LABEL_ORDER))
    confusion = confusion_matrix(y_true, pred_labels, labels=LABEL_ORDER)

    try:
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=LABEL_ORDER)
        roc_auc = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc = float("nan")

    structure_edges = sorted((str(src), str(dst)) for src, dst in model.edges())
    bundle = {
        "model": model,
        "features": feature_cols,
        "discretizers": discretizers,
        "label_order": LABEL_ORDER,
        "target_col": TARGET_COL,
        "config": {
            "bins": args.bins,
            "max_indegree": args.max_indegree,
            "equivalent_sample_size": args.equivalent_sample_size,
            "test_frac": args.test_frac,
            "max_fixtures": args.max_fixtures,
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "roc_auc_weighted_ovr": roc_auc,
        },
        "structure_edges": structure_edges,
    }

    model_path = MODEL_DIR / "pretrain_model_bayesian_strict.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    report_lines = [
        "=" * 72,
        "STRICT NO-LEAKAGE BAYESIAN NETWORK REPORT - PRE-MATCH",
        "=" * 72,
        "",
        f"Data file: {DATA_FILE}",
        f"Train/Test samples: {len(train_df)}/{len(test_df)}",
        f"Train/Test fixtures: {train_df['fixture_id'].nunique()}/{test_df['fixture_id'].nunique()}",
        f"Feature count: {len(feature_cols)}",
        f"Discretization bins: {args.bins}",
        f"Max in-degree: {args.max_indegree}",
        f"Equivalent sample size: {args.equivalent_sample_size}",
        f"Temporal test fraction: {args.test_frac}",
        "Method note: Bayesian network inspired by the 2017 paper, but restricted to true pre-match features only.",
        "",
        f"Accuracy: {accuracy:.4f}",
        f"Macro-F1: {macro_f1:.4f}",
        f"ROC-AUC (weighted ovr): {roc_auc:.4f}",
        "",
        "Classification report:",
        classification_report(y_true, pred_labels, target_names=["Away Win", "Draw", "Home Win"], labels=LABEL_ORDER, zero_division=0),
        "Confusion matrix [A, D, H]:",
        str(confusion),
        "",
        "Selected features:",
        ", ".join(feature_cols),
        "",
        f"Learned structure edge count: {len(structure_edges)}",
        "Learned edges (first 60):",
        str(structure_edges[:60]),
        "",
        f"Model saved to: {model_path}",
    ]

    report_path = REPORT_DIR / "pretrain_report_bayesian_strict.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Strict no-leakage Bayesian pre-match model")
    print("=" * 72)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"ROC-AUC (weighted ovr): {roc_auc:.4f}")
    print(f"Learned edge count: {len(structure_edges)}")
    print(f"Model saved:  {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()