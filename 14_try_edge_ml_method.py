"""
Step 14: Reproduce-style attempt of edge-computing + ML football prediction

This script implements a practical approximation of the paper method:
- Multi-time-scale feature extraction with 8 simulated edge nodes
  - Common features on windows: 3, 4, 5, 6 matches
  - Contrastive (head-to-head) features on windows: 3, 4, 5, 6 matches
- Three classifiers:
  - KNN
  - Logistic Regression
  - BP-style Neural Network (MLP)
- Cloud fusion by averaging class probabilities from all edge nodes
- Temporal holdout evaluation (last 20% fixtures by date)

Inputs:
- data/processed/pretrain_dataset.csv

Outputs:
- reports/edge_ml_method_report.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

DATA_FILE = Path("data/processed/pretrain_dataset.csv")
REPORT_FILE = Path("reports/edge_ml_method_report.txt")
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

WINDOWS = [3, 4, 5, 6]
LABEL_ORDER = ["A", "D", "H"]

# Keep output clean while preserving actual runtime errors.
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"sklearn\.neural_network\._multilayer_perceptron",
)


@dataclass
class NodeData:
    name: str
    feature_cols: List[str]


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _result_from_home_perspective(row: pd.Series) -> str:
    return str(row["result"])


def _team_result_from_match(team: str, row: pd.Series) -> str:
    home = row["home_team"]
    away = row["away_team"]
    r = _result_from_home_perspective(row)
    if team == home:
        if r == "H":
            return "W"
        if r == "A":
            return "L"
        return "D"
    if team == away:
        if r == "A":
            return "W"
        if r == "H":
            return "L"
        return "D"
    return "D"


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _summarize_team_history(history: List[Dict], window: int) -> Dict[str, float]:
    recent = history[-window:]
    if not recent:
        return {
            "games": 0.0,
            "ppg": 0.0,
            "gf_avg": 0.0,
            "ga_avg": 0.0,
            "gd_avg": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
        }

    wins = sum(1 for x in recent if x["result"] == "W")
    draws = sum(1 for x in recent if x["result"] == "D")
    losses = sum(1 for x in recent if x["result"] == "L")

    points = wins * 3 + draws
    gf = sum(x["gf"] for x in recent)
    ga = sum(x["ga"] for x in recent)
    n = float(len(recent))

    return {
        "games": n,
        "ppg": points / n,
        "gf_avg": gf / n,
        "ga_avg": ga / n,
        "gd_avg": (gf - ga) / n,
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
    }


def _summarize_h2h_history(history: List[Dict], home_team: str, away_team: str, window: int) -> Dict[str, float]:
    recent = history[-window:]
    if not recent:
        return {
            "h2h_games": 0.0,
            "h2h_home_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_home_gd_avg": 0.0,
        }

    home_wins = 0
    draws = 0
    gds = []

    for m in recent:
        if m["home_team"] == home_team and m["away_team"] == away_team:
            r = m["result"]
            gd = m["goals_home"] - m["goals_away"]
        else:
            # same pair but reversed home/away
            raw_r = m["result"]
            if raw_r == "H":
                r = "A"
            elif raw_r == "A":
                r = "H"
            else:
                r = "D"
            gd = m["goals_away"] - m["goals_home"]

        if r == "H":
            home_wins += 1
        elif r == "D":
            draws += 1
        gds.append(float(gd))

    n = float(len(recent))
    return {
        "h2h_games": n,
        "h2h_home_win_rate": home_wins / n,
        "h2h_draw_rate": draws / n,
        "h2h_home_gd_avg": _safe_mean(gds),
    }


def build_edge_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[NodeData]]:
    data = df.copy()
    data["date"] = _to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    team_history: Dict[str, List[Dict]] = {}
    h2h_history: Dict[Tuple[str, str], List[Dict]] = {}

    node_defs: List[NodeData] = []
    all_rows: List[Dict[str, float]] = []

    for window in WINDOWS:
        node_defs.append(
            NodeData(
                name=f"common_w{window}",
                feature_cols=[
                    f"w{window}_diff_ppg",
                    f"w{window}_diff_gf_avg",
                    f"w{window}_diff_ga_avg",
                    f"w{window}_diff_gd_avg",
                    f"w{window}_diff_win_rate",
                    f"w{window}_diff_draw_rate",
                    f"w{window}_diff_loss_rate",
                    f"w{window}_home_games",
                    f"w{window}_away_games",
                ],
            )
        )
        node_defs.append(
            NodeData(
                name=f"contrastive_w{window}",
                feature_cols=[
                    f"h2h_w{window}_games",
                    f"h2h_w{window}_home_win_rate",
                    f"h2h_w{window}_draw_rate",
                    f"h2h_w{window}_home_gd_avg",
                ],
            )
        )

    for _, row in data.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []

        pair_key = tuple(sorted([home, away]))
        if pair_key not in h2h_history:
            h2h_history[pair_key] = []

        features: Dict[str, float] = {}

        for window in WINDOWS:
            hs = _summarize_team_history(team_history[home], window)
            as_ = _summarize_team_history(team_history[away], window)
            features[f"w{window}_diff_ppg"] = hs["ppg"] - as_["ppg"]
            features[f"w{window}_diff_gf_avg"] = hs["gf_avg"] - as_["gf_avg"]
            features[f"w{window}_diff_ga_avg"] = hs["ga_avg"] - as_["ga_avg"]
            features[f"w{window}_diff_gd_avg"] = hs["gd_avg"] - as_["gd_avg"]
            features[f"w{window}_diff_win_rate"] = hs["win_rate"] - as_["win_rate"]
            features[f"w{window}_diff_draw_rate"] = hs["draw_rate"] - as_["draw_rate"]
            features[f"w{window}_diff_loss_rate"] = hs["loss_rate"] - as_["loss_rate"]
            features[f"w{window}_home_games"] = hs["games"]
            features[f"w{window}_away_games"] = as_["games"]

            h2h = _summarize_h2h_history(h2h_history[pair_key], home, away, window)
            features[f"h2h_w{window}_games"] = h2h["h2h_games"]
            features[f"h2h_w{window}_home_win_rate"] = h2h["h2h_home_win_rate"]
            features[f"h2h_w{window}_draw_rate"] = h2h["h2h_draw_rate"]
            features[f"h2h_w{window}_home_gd_avg"] = h2h["h2h_home_gd_avg"]

        features["date"] = row["date"]
        features["fixture_id"] = row["fixture_id"]
        features["result"] = row["result"]
        all_rows.append(features)

        # Update history after feature extraction to avoid future leakage.
        team_history[home].append(
            {
                "result": _team_result_from_match(home, row),
                "gf": float(row["goals_home"]),
                "ga": float(row["goals_away"]),
            }
        )
        team_history[away].append(
            {
                "result": _team_result_from_match(away, row),
                "gf": float(row["goals_away"]),
                "ga": float(row["goals_home"]),
            }
        )

        h2h_history[pair_key].append(
            {
                "home_team": home,
                "away_team": away,
                "goals_home": float(row["goals_home"]),
                "goals_away": float(row["goals_away"]),
                "result": str(row["result"]),
            }
        )

    feat_df = pd.DataFrame(all_rows)
    return feat_df, node_defs


def temporal_holdout_mask(df: pd.DataFrame, frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    ordered = df.sort_values("date")
    n_test = max(1, int(len(ordered) * frac))
    test_idx = ordered.tail(n_test).index
    test_mask = df.index.isin(test_idx)
    train_mask = ~test_mask
    return np.asarray(train_mask), np.asarray(test_mask)


def build_model(kind: str):
    if kind == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=11, weights="distance")),
        ])
    if kind == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                ),
            ),
        ])
    if kind == "bp":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    learning_rate_init=1e-3,
                    alpha=1e-4,
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ])
    raise ValueError(f"Unknown model kind: {kind}")


def run_fusion_experiment(df_feat: pd.DataFrame, node_defs: List[NodeData], model_kind: str) -> Dict[str, object]:
    df_feat = df_feat.copy().reset_index(drop=True)
    train_mask, test_mask = temporal_holdout_mask(df_feat, frac=0.2)

    y_train = df_feat.loc[train_mask, "result"].to_numpy()
    y_test = df_feat.loc[test_mask, "result"].to_numpy()

    n_classes = len(LABEL_ORDER)
    proba_sum = np.zeros((len(y_test), n_classes), dtype=float)
    valid_nodes = 0

    per_node_acc = {}

    for node in node_defs:
        X_train = df_feat.loc[train_mask, node.feature_cols].fillna(0.0).to_numpy()
        X_test = df_feat.loc[test_mask, node.feature_cols].fillna(0.0).to_numpy()

        model = build_model(model_kind)
        model.fit(X_train, y_train)

        node_proba = model.predict_proba(X_test)
        proba_sum += node_proba
        valid_nodes += 1

        node_pred = model.predict(X_test)
        per_node_acc[node.name] = float(accuracy_score(y_test, node_pred))

    fused_proba = proba_sum / max(valid_nodes, 1)
    pred_idx = fused_proba.argmax(axis=1)
    y_pred = np.array([LABEL_ORDER[i] for i in pred_idx])

    acc = float(accuracy_score(y_test, y_pred))
    cls = classification_report(y_test, y_pred, labels=LABEL_ORDER, digits=4)

    return {
        "model": model_kind,
        "accuracy": acc,
        "classification_report": cls,
        "per_node_acc": per_node_acc,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")

    raw = pd.read_csv(DATA_FILE)
    required = {"fixture_id", "date", "home_team", "away_team", "result", "goals_home", "goals_away"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    print("Building edge-style features...")
    feat_df, nodes = build_edge_features(raw)

    results = []
    for kind in ["knn", "lr", "bp"]:
        print(f"Running fusion model: {kind}")
        out = run_fusion_experiment(feat_df, nodes, kind)
        results.append(out)
        print(f"  accuracy={out['accuracy']:.4f} | train={out['n_train']} test={out['n_test']}")

    best = max(results, key=lambda x: x["accuracy"])

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("EDGE COMPUTING + ML METHOD (REPRODUCTION-STYLE ATTEMPT)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Data file: {DATA_FILE}")
    lines.append(f"Total fixtures: {len(feat_df)}")
    lines.append(f"Node count: {len(nodes)}")
    lines.append("Nodes:")
    for n in nodes:
        lines.append(f"  - {n.name}: {len(n.feature_cols)} features")
    lines.append("")

    for res in results:
        lines.append("-" * 72)
        lines.append(f"Model: {res['model']}")
        lines.append(f"Temporal split train/test: {res['n_train']}/{res['n_test']}")
        lines.append(f"Fused accuracy: {res['accuracy']:.4f}")
        lines.append("Per-node accuracy:")
        for name, acc in sorted(res["per_node_acc"].items()):
            lines.append(f"  {name:<18} {acc:.4f}")
        lines.append("")
        lines.append("Classification report (labels A/D/H):")
        lines.append(str(res["classification_report"]))
        lines.append("")

    lines.append("=" * 72)
    lines.append(f"Best fused model: {best['model']} | accuracy={best['accuracy']:.4f}")
    lines.append("=" * 72)

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
