"""
Step 4: Prediction Demo
Demonstrates how to use the two trained models for pre-match and in-play predictions
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

print("Loading trained models...")

with open(MODEL_DIR / "pretrain_model.pkl", "rb") as f:
    pretrain_data = pickle.load(f)
    pretrain_model = pretrain_data['model']
    pretrain_features = pretrain_data['features']
    pretrain_scaler = pretrain_data['scaler']
    pretrain_label_encoder = pretrain_data.get('label_encoder')

with open(MODEL_DIR / "inplay_model.pkl", "rb") as f:
    inplay_data = pickle.load(f)
    inplay_model = inplay_data['model']
    inplay_features = inplay_data['features']
    inplay_scaler = inplay_data['scaler']

print("✓ Models loaded successfully")


pretrain_df = pd.read_csv(DATA_DIR / "pretrain_dataset.csv")
inplay_df = pd.read_csv(DATA_DIR / "inplay_dataset.csv")


sample_fixtures = pretrain_df[pretrain_df['h_games_played'] >= 20].head(5)

print("\n" + "="*70)
print("DEMO 1: PRE-MATCH PREDICTION")
print("="*70)

for idx, row in sample_fixtures.iterrows():
    X_sample = row[pretrain_features].to_frame().T  
    if pretrain_scaler is not None:
        X_sample = pretrain_scaler.transform(X_sample)
    

    pred_class = pretrain_model.predict(X_sample)[0]
    pred_proba = pretrain_model.predict_proba(X_sample)[0]
    if pretrain_label_encoder is not None:
        pred_class = pretrain_label_encoder.inverse_transform([int(pred_class)])[0]
    

    class_order = ['A', 'D', 'H']
    if pretrain_label_encoder is not None:
        class_order = list(pretrain_label_encoder.classes_)

    prob_lookup = {label: pred_proba[idx] for idx, label in enumerate(class_order)}
    prob_dict = {
        'Away': prob_lookup.get('A', 0.0),
        'Draw': prob_lookup.get('D', 0.0),
        'Home': prob_lookup.get('H', 0.0),
    }
    
    print(f"\nMatch: {row['home_team']} vs {row['away_team']}")
    print(f"Date: {row['date']}")
    print(f"Formations: {row['home_formation']} vs {row['away_formation']}")
    print(f"Home form (last 5): W:{row['h_recent_wins']} D:{row['h_recent_draws']} L:{row['h_recent_losses']} | Avg GF:{row['h_avg_gf']:.2f} GA:{row['h_avg_ga']:.2f}")
    print(f"Away form (last 5): W:{row['a_recent_wins']} D:{row['a_recent_draws']} L:{row['a_recent_losses']} | Avg GF:{row['a_avg_gf']:.2f} GA:{row['a_avg_ga']:.2f}")
    
    print(f"\nPrediction: {pred_class}")
    print(f"Probabilities:")
    print(f"  Away Win: {prob_dict['Away']:.1%}")
    print(f"  Draw:     {prob_dict['Draw']:.1%}")
    print(f"  Home Win: {prob_dict['Home']:.1%}")
    print(f"Actual Result: {row['result']} ({row['goals_home']}-{row['goals_away']})")



print("\n" + "="*70)
print("DEMO 2: IN-PLAY PREDICTION")
print("="*70)


interesting = pretrain_df[
    (pretrain_df['goals_home'] + pretrain_df['goals_away'] >= 4) &
    (pretrain_df['h_games_played'] >= 20)
]
sample_fixture_id = interesting.iloc[0]['fixture_id'] if len(interesting) > 0 else pretrain_df.iloc[0]['fixture_id']
match_inplay = inplay_df[inplay_df['fixture_id'] == sample_fixture_id].sort_values('minute')

if len(match_inplay) > 0:
    match_info = pretrain_df[pretrain_df['fixture_id'] == sample_fixture_id].iloc[0]
    print(f"\nMatch: {match_info['home_team']} vs {match_info['away_team']}")
    print(f"Final Score: {match_info['goals_home']}-{match_info['goals_away']} ({match_info['result']})")
    print(f"\nPredictions at different match minutes:\n")
    print(f"{'Min':<4} {'HG':<3} {'AG':<3} {'YH':<3} {'YA':<3} {'Prediction':<12} {'Probabilities':<50}")
    print("-" * 80)
    
    for _, row in match_inplay.iterrows():
        X_sample = row[inplay_features].to_frame().T
        
        if inplay_scaler is not None:
            X_sample = inplay_scaler.transform(X_sample)
        
        pred_class = inplay_model.predict(X_sample)[0]
        pred_proba = inplay_model.predict_proba(X_sample)[0]
        
        prob_str = f"A:{pred_proba[0]:.0%} D:{pred_proba[1]:.0%} H:{pred_proba[2]:.0%}"
        
        print(f"{int(row['minute']):<4} {int(row['goals_home']):<3} {int(row['goals_away']):<3} "
              f"{int(row['yellow_home']):<3} {int(row['yellow_away']):<3} {pred_class:<12} {prob_str:<50}")


print("\n" + "="*70)
print("DEMO 3: MODEL COMPARISON ANALYSIS")
print("="*70)

from sklearn.metrics import accuracy_score


pretrain_df_sorted = pretrain_df.sort_values('date')
pretrain_test = pretrain_df_sorted.tail(int(len(pretrain_df_sorted) * 0.2))
X_pretrain_test = pretrain_test[pretrain_features]  
y_pretrain_test = pretrain_test['result'].values

if pretrain_scaler is not None:
    X_pretrain_test = pretrain_scaler.transform(X_pretrain_test)

pretrain_pred = pretrain_model.predict(X_pretrain_test)
if pretrain_label_encoder is not None:
    pretrain_pred = pretrain_label_encoder.inverse_transform(pretrain_pred.astype(int))
pretrain_acc = accuracy_score(y_pretrain_test, pretrain_pred)

print(f"\nPre-match Model Performance (on last-20% by date):")
print(f"  Accuracy: {pretrain_acc:.4f}")


inplay_df_sorted = inplay_df.sort_values('date')
inplay_test = inplay_df_sorted.tail(int(len(inplay_df_sorted) * 0.2))
X_inplay_test = inplay_test[inplay_features]  
y_inplay_test = inplay_test['result'].values

if inplay_scaler is not None:
    X_inplay_test = inplay_scaler.transform(X_inplay_test)

inplay_pred = inplay_model.predict(X_inplay_test)
inplay_acc = accuracy_score(y_inplay_test, inplay_pred)

print(f"\nIn-play Model Performance (on last-20% by date):")
print(f"  Accuracy: {inplay_acc:.4f}")


print(f"\nIn-play Accuracy by Minute (predicting final result H/D/A):")
for minute in [15, 30, 45, 60, 75, 90]:
    minute_data = inplay_test[inplay_test['minute'] == minute]
    if len(minute_data) > 0:
        X_minute = minute_data[inplay_features]  
        y_minute = minute_data['result'].values
        
        if inplay_scaler is not None:
            X_minute = inplay_scaler.transform(X_minute)
        
        minute_pred = inplay_model.predict(X_minute)
        minute_acc = accuracy_score(y_minute, minute_pred)
        print(f"  {minute:2d}': {minute_acc:.4f}")

print("\n" + "="*70)
print("✓ Demo complete!")
print("="*70)
