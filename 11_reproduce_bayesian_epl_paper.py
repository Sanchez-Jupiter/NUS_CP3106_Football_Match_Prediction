"""
Reproduce the 2017 EPL Bayesian-network paper in a controlled way.

Paper:
Predicting Football Matches Results using Bayesian Networks for English Premier League (2017)

What this script does:
1. Downloads EPL CSV files for seasons 2010-2011, 2011-2012, 2012-2013 from football-data.co.uk.
2. Builds a Bayesian-network classifier using pgmpy.
3. Evaluates season by season with K-fold cross validation, matching the paper's setup as closely as practical.
4. Reports two experimental views:
   - paper_like: uses the match-summary attributes shown in the paper table.
   - paper_like_with_ft_goals: additionally includes full-time goals, which is likely label leakage.

Important caveat:
Most attributes listed in the paper table are not true pre-match features. Shots, shots on target,
corners, fouls, cards, and halftime goals are only known during or after the match. The reported
75.09% accuracy is therefore not directly comparable to a strict pre-match prediction task.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianModel

try:
    from pgmpy.estimators import BDeu
except ImportError:
    from pgmpy.estimators import BDeuScore as BDeu


ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "football_data_uk"
REPORT_DIR = ROOT_DIR / "reports"
RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_COL = "FTR"
LABEL_ORDER = ["A", "D", "H"]
SEASON_URLS = {
    "2010-2011": "https://www.football-data.co.uk/mmz4281/1011/E0.csv",
    "2011-2012": "https://www.football-data.co.uk/mmz4281/1112/E0.csv",
    "2012-2013": "https://www.football-data.co.uk/mmz4281/1213/E0.csv",
}

PAPER_TABLE_FEATURES = [
    "HomeTeam",
    "AwayTeam",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    "HTHG",
    "HTAG",
]


@dataclass
class ExperimentResult:
    season: str
    feature_mode: str
    accuracy: float
    macro_f1: float
    support: int
    confusion: list[list[int]]


def _load_or_download_season(season: str) -> pd.DataFrame:
    csv_path = RAW_DIR / f"epl_{season}.csv"
    if not csv_path.exists():
        df = pd.read_csv(SEASON_URLS[season])
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df


def _prepare_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    required_cols = feature_cols + [TARGET_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = df[required_cols].copy()
    prepared = prepared.dropna(subset=[TARGET_COL])
    prepared[TARGET_COL] = prepared[TARGET_COL].astype(str)
    return prepared


def _discretize_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=train_df.index)
    test_out = pd.DataFrame(index=test_df.index)

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            train_values = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
            test_values = pd.to_numeric(test_df[col], errors="coerce").fillna(0.0)

            unique_values = np.unique(train_values.to_numpy())
            if len(unique_values) <= bins:
                mapping = {value: f"v_{idx}" for idx, value in enumerate(sorted(unique_values.tolist()))}
                sorted_values = np.array(sorted(unique_values.tolist()), dtype=float)
                train_out[col] = train_values.map(mapping).astype(str)
                test_out[col] = test_values.map(
                    lambda value: mapping.get(value, mapping[sorted_values[np.argmin(np.abs(sorted_values - value))]])
                ).astype(str)
            else:
                quantiles = np.linspace(0, 1, bins + 1)
                edges = np.quantile(train_values, quantiles)
                edges = np.unique(edges)
                if len(edges) <= 2:
                    train_out[col] = "bin_0"
                    test_out[col] = "bin_0"
                else:
                    edges = edges.astype(float)
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    train_bins = pd.cut(train_values, bins=edges, include_lowest=True, duplicates="drop")
                    test_bins = pd.cut(test_values, bins=edges, include_lowest=True, duplicates="drop")
                    train_out[col] = train_bins.astype(str)
                    test_out[col] = test_bins.astype(str)
        else:
            train_out[col] = train_df[col].fillna("missing").astype(str)
            test_out[col] = test_df[col].fillna("missing").astype(str)

    train_out[TARGET_COL] = train_df[TARGET_COL].astype(str)
    test_out[TARGET_COL] = test_df[TARGET_COL].astype(str)
    return train_out, test_out


def _fit_bn(train_df: pd.DataFrame, feature_cols: list[str]) -> BayesianModel:
    search = HillClimbSearch(train_df)
    learned = search.estimate(scoring_method=BDeu(train_df), max_indegree=3, show_progress=False)

    edges = list(learned.edges()) if hasattr(learned, "edges") else list(learned)
    model = BayesianModel(edges)

    for col in feature_cols + [TARGET_COL]:
        if col not in model.nodes():
            model.add_node(col)

    if TARGET_COL not in model.nodes():
        model.add_node(TARGET_COL)

    model.fit(
        train_df,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
    )
    return model


def _predict_bn(model: BayesianModel, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    infer = VariableElimination(model)
    predictions: list[str] = []

    for _, row in test_df.iterrows():
        evidence = {col: str(row[col]) for col in feature_cols}
        query = infer.query(variables=[TARGET_COL], evidence=evidence, show_progress=False)
        state_names = query.state_names[TARGET_COL]
        pred_idx = int(np.argmax(query.values))
        predictions.append(str(state_names[pred_idx]))

    return np.asarray(predictions, dtype=object)


def _run_cv_for_season(season: str, feature_cols: list[str], n_splits: int) -> ExperimentResult:
    season_df = _load_or_download_season(season)
    season_df = _prepare_columns(season_df, feature_cols)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    all_true: list[str] = []
    all_pred: list[str] = []

    for train_idx, test_idx in splitter.split(season_df):
        train_df = season_df.iloc[train_idx].reset_index(drop=True)
        test_df = season_df.iloc[test_idx].reset_index(drop=True)
        train_disc, test_disc = _discretize_fold(train_df, test_df, feature_cols)

        model = _fit_bn(train_disc, feature_cols)
        pred = _predict_bn(model, test_disc, feature_cols)
        all_true.extend(test_disc[TARGET_COL].tolist())
        all_pred.extend(pred.tolist())

    y_true = np.asarray(all_true, dtype=object)
    y_pred = np.asarray(all_pred, dtype=object)
    return ExperimentResult(
        season=season,
        feature_mode="",
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER)),
        support=int(len(y_true)),
        confusion=confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
    )


def _format_result_block(title: str, results: Iterable[ExperimentResult]) -> str:
    lines = ["=" * 72, title, "=" * 72, ""]
    accs = []
    f1s = []
    supports = []

    for result in results:
        accs.append(result.accuracy)
        f1s.append(result.macro_f1)
        supports.append(result.support)
        lines.append(f"Season: {result.season}")
        lines.append(f"Accuracy: {result.accuracy:.4f}")
        lines.append(f"Macro-F1: {result.macro_f1:.4f}")
        lines.append(f"Support: {result.support}")
        lines.append("Confusion matrix [A, D, H]:")
        lines.append(str(np.asarray(result.confusion)))
        lines.append("")

    weighted_acc = float(np.average(accs, weights=supports))
    weighted_f1 = float(np.average(f1s, weights=supports))
    lines.append(f"Weighted average accuracy: {weighted_acc:.4f}")
    lines.append(f"Weighted average macro-F1: {weighted_f1:.4f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the 2017 EPL Bayesian-network paper.")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds per season.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASON_URLS.keys()),
        choices=list(SEASON_URLS.keys()),
        help="Subset of seasons to evaluate.",
    )
    parser.add_argument(
        "--include-fulltime-goals",
        action="store_true",
        help="Include FTHG/FTAG from the paper table. This is direct leakage and only for comparison.",
    )
    args = parser.parse_args()

    paper_like_features = PAPER_TABLE_FEATURES.copy()
    experiment_name = "paper_like"
    if args.include_fulltime_goals:
        paper_like_features += ["FTHG", "FTAG"]
        experiment_name = "paper_like_with_ft_goals"

    results: list[ExperimentResult] = []
    for season in args.seasons:
        result = _run_cv_for_season(season, paper_like_features, args.folds)
        result.feature_mode = experiment_name
        results.append(result)

    report_text = _format_result_block(
        f"BAYESIAN NETWORK EPL PAPER REPRODUCTION - {experiment_name.upper()}",
        results,
    )
    report_text += "\n" + "Notes:\n"
    report_text += "- The paper's reported 75.09% average accuracy uses match-summary attributes, not strict pre-match features.\n"
    report_text += "- HS/HST/HC/HF/HY/HR and their away-team counterparts are only known after the match progresses.\n"
    report_text += "- HTHG/HTAG are halftime features, still not pre-match.\n"
    if args.include_fulltime_goals:
        report_text += "- FTHG/FTAG are direct label leakage and should only be used to test whether the paper setup is leaky.\n"

    report_path = REPORT_DIR / f"paper_bayesian_epl_{experiment_name}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    summary = {
        "experiment": experiment_name,
        "folds": args.folds,
        "seasons": args.seasons,
        "features": paper_like_features,
        "results": [result.__dict__ for result in results],
    }
    summary_path = REPORT_DIR / f"paper_bayesian_epl_{experiment_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report_text)
    print(f"\nReport saved to: {report_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()