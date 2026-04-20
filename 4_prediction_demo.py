"""
Step 4: Prediction Demo
Demonstrates how to use the two trained models for pre-match and in-play predictions.

Usage:
    python 4_prediction_demo.py

Requires:
    - models/pretrain_model.pkl  (from step 2)
    - models/inplay_model.pkl    (from step 3)
    - data/processed/pretrain_dataset.csv
    - data/processed/inplay_dataset.csv
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

# ── Load models ──────────────────────────────────────────────
print("Loading trained models...")

with open(MODEL_DIR / "pretrain_model.pkl", "rb") as f:
    pretrain_data = pickle.load(f)
    pretrain_model = pretrain_data['model']
    pretrain_features = pretrain_data['features']
    pretrain_scaler = pretrain_data['scaler']
    pretrain_label_encoder = pretrain_data.get('label_encoder')
    pretrain_model_name = pretrain_data.get('model_name', 'Unknown')

with open(MODEL_DIR / "inplay_model.pkl", "rb") as f:
    inplay_data = pickle.load(f)
    inplay_model = inplay_data['model']
    inplay_features = inplay_data['features']
    inplay_scaler = inplay_data['scaler']
    inplay_model_name = inplay_data.get('model_name', 'Unknown')

print(f"✓ Pre-match model loaded: {pretrain_model_name}")
print(f"✓ In-play  model loaded: {inplay_model_name}")

# ── Load datasets ────────────────────────────────────────────
pretrain_df = pd.read_csv(DATA_DIR / "pretrain_dataset.csv")
inplay_df = pd.read_csv(DATA_DIR / "inplay_dataset.csv")
print(f"✓ Pre-match dataset: {len(pretrain_df):,} fixtures")
print(f"✓ In-play  dataset: {len(inplay_df):,} rows ({inplay_df['fixture_id'].nunique():,} fixtures)")

# Pick fixtures with sufficient history for a meaningful demo
sample_fixtures = pretrain_df[pretrain_df['h_games_played'] >= 20].tail(5)

print("\n" + "="*70)
print("DEMO 1: PRE-MATCH PREDICTION")
print("="*70)

def get_prob_dict(model, X, label_encoder=None):
    """Get prediction and probability dict {H/D/A: prob} from a model."""
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    if label_encoder is not None:
        pred = label_encoder.inverse_transform([int(pred)])[0]
        classes = list(label_encoder.classes_)
    else:
        classes = list(model.classes_)
    prob_map = {cls: proba[i] for i, cls in enumerate(classes)}
    return pred, prob_map


RESULT_LABELS = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}

for _, row in sample_fixtures.iterrows():
    X_sample = row[pretrain_features].to_frame().T
    if pretrain_scaler is not None:
        X_sample = pretrain_scaler.transform(X_sample)

    pred_class, prob_map = get_prob_dict(pretrain_model, X_sample, pretrain_label_encoder)

    print(f"\nMatch: {row['home_team']} vs {row['away_team']}")
    print(f"Date: {row['date']}")
    print(f"Formations: {row['home_formation']} vs {row['away_formation']}")
    print(f"Home form (last 5): W:{row['h_recent_wins']} D:{row['h_recent_draws']} L:{row['h_recent_losses']} | Avg GF:{row['h_avg_gf']:.2f} GA:{row['h_avg_ga']:.2f}")
    print(f"Away form (last 5): W:{row['a_recent_wins']} D:{row['a_recent_draws']} L:{row['a_recent_losses']} | Avg GF:{row['a_avg_gf']:.2f} GA:{row['a_avg_ga']:.2f}")

    correct = '✓' if pred_class == row['result'] else '✗'
    print(f"\nPrediction: {RESULT_LABELS.get(pred_class, pred_class)}")
    print(f"Probabilities:  Home {prob_map.get('H', 0):.1%}  |  Draw {prob_map.get('D', 0):.1%}  |  Away {prob_map.get('A', 0):.1%}")
    print(f"Actual Result:  {RESULT_LABELS.get(row['result'], row['result'])} ({int(row['goals_home'])}-{int(row['goals_away'])})  {correct}")



print("\n" + "="*70)
print("DEMO 2: IN-PLAY PREDICTION")
print("="*70)

# Pick a high-scoring match that also exists in the in-play dataset
inplay_fixture_ids = set(inplay_df['fixture_id'].unique())
interesting = pretrain_df[
    (pretrain_df['goals_home'] + pretrain_df['goals_away'] >= 4) &
    (pretrain_df['h_games_played'] >= 20) &
    (pretrain_df['fixture_id'].isin(inplay_fixture_ids))
]
if len(interesting) == 0:
    interesting = pretrain_df[pretrain_df['fixture_id'].isin(inplay_fixture_ids)]

sample_fixture_id = interesting.iloc[0]['fixture_id']
match_inplay = inplay_df[inplay_df['fixture_id'] == sample_fixture_id].sort_values('minute')

if len(match_inplay) > 0:
    match_info = pretrain_df[pretrain_df['fixture_id'] == sample_fixture_id].iloc[0]
    print(f"\nMatch: {match_info['home_team']} vs {match_info['away_team']}")
    print(f"Final Score: {int(match_info['goals_home'])}-{int(match_info['goals_away'])} ({RESULT_LABELS.get(match_info['result'], match_info['result'])})")
    print(f"\nMinute-by-minute predictions:\n")
    print(f"{'Min':<5} {'Score':<7} {'Cards':<8} {'Prediction':<12} {'P(Home)':<9} {'P(Draw)':<9} {'P(Away)':<9}")
    print("-" * 65)

    for _, row in match_inplay.iterrows():
        X_sample = row[inplay_features].to_frame().T
        if inplay_scaler is not None:
            X_sample = inplay_scaler.transform(X_sample)

        pred_class, prob_map = get_prob_dict(inplay_model, X_sample)

        score_str = f"{int(row['goals_home'])}-{int(row['goals_away'])}"
        card_str = f"Y:{int(row['yellow_home'])}-{int(row['yellow_away'])}"
        if row.get('red_home', 0) + row.get('red_away', 0) > 0:
            card_str += f" R:{int(row['red_home'])}-{int(row['red_away'])}"

        print(f"{int(row['minute']):<5} {score_str:<7} {card_str:<8} "
              f"{RESULT_LABELS.get(pred_class, pred_class):<12} "
              f"{prob_map.get('H', 0):<9.1%} {prob_map.get('D', 0):<9.1%} {prob_map.get('A', 0):<9.1%}")
else:
    print("\n(No in-play data found for the selected fixture.)")


print("\n" + "="*70)
print("DEMO 3: MODEL EVALUATION")
print("="*70)

# ── Pre-match evaluation (temporal split: last 20% by date) ──
pretrain_df_sorted = pretrain_df.sort_values('date')
split_idx = int(len(pretrain_df_sorted) * 0.8)
pretrain_test = pretrain_df_sorted.iloc[split_idx:]
X_pretrain_test = pretrain_test[pretrain_features]
y_pretrain_test = pretrain_test['result'].values

if pretrain_scaler is not None:
    X_pretrain_test = pretrain_scaler.transform(X_pretrain_test)

pretrain_pred = pretrain_model.predict(X_pretrain_test)
if pretrain_label_encoder is not None:
    pretrain_pred = pretrain_label_encoder.inverse_transform(pretrain_pred.astype(int))

pretrain_acc = accuracy_score(y_pretrain_test, pretrain_pred)
print(f"\nPre-match Model ({pretrain_model_name}):")
print(f"  Test fixtures: {len(pretrain_test):,}")
print(f"  Accuracy:      {pretrain_acc:.4f}")
print(f"  Baseline (always Home): {(y_pretrain_test == 'H').mean():.4f}")

# ── In-play evaluation ──
inplay_df_sorted = inplay_df.sort_values('date')
split_idx_ip = int(len(inplay_df_sorted) * 0.8)
inplay_test = inplay_df_sorted.iloc[split_idx_ip:]
X_inplay_test = inplay_test[inplay_features]
y_inplay_test = inplay_test['result'].values

if inplay_scaler is not None:
    X_inplay_test = inplay_scaler.transform(X_inplay_test)

inplay_pred = inplay_model.predict(X_inplay_test)
inplay_acc = accuracy_score(y_inplay_test, inplay_pred)

print(f"\nIn-play Model ({inplay_model_name}):")
print(f"  Test rows:  {len(inplay_test):,}")
print(f"  Accuracy:   {inplay_acc:.4f}")

# ── Accuracy by minute (use actual checkpoint intervals) ──
available_minutes = sorted(inplay_test['minute'].unique())
print(f"\nIn-play Accuracy by Minute:")
print(f"  {'Min':<6} {'Acc':<8} {'N':>6}")
print(f"  {'-'*22}")
for minute in available_minutes:
    minute_data = inplay_test[inplay_test['minute'] == minute]
    if len(minute_data) > 0:
        X_minute = minute_data[inplay_features]
        y_minute = minute_data['result'].values
        if inplay_scaler is not None:
            X_minute = inplay_scaler.transform(X_minute)
        minute_pred = inplay_model.predict(X_minute)
        minute_acc = accuracy_score(y_minute, minute_pred)
        print(f"  {int(minute):>3}'   {minute_acc:.4f}   {len(minute_data):>5}")

print("\n" + "="*70)
print("✓ Demo complete!")
print("="*70)
