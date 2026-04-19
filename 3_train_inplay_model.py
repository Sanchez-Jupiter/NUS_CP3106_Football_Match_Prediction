"""
Step 3: In-play 15-minute interval prediction model training
Input: data/processed/inplay_dataset.csv
Output:
  - models/inplay_model.pkl  (Trained model)
  - reports/inplay_report.txt (Performance evaluation)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = Path("data/processed/inplay_dataset.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading in-play dataset...")
df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nLabel distribution:")
print(df['result'].value_counts())

# Feature selection and processing
# Automatically select numeric features, excluding labels and obvious leakage fields
exclude_cols = {
    'fixture_id', 'date', 'home_team', 'away_team',
    'result', 'ft_home', 'ft_away'
}
candidate_cols = [c for c in df.columns if c not in exclude_cols]
feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

X = df[feature_cols].copy()
y = df['result'].copy()

# Handle missing values
X = X.fillna(0)

print(f"\nFeatures shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")
print(f"\nFeature statistics:")
print(X.describe())

# Data splitting and scaling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Model training: Compare two models
# ============================================================

print("\n" + "="*60)
print("Training Model 1: Random Forest")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_acc:.4f}")

print("\n" + "="*60)
print("Training Model 2: Gradient Boosting")
print("="*60)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
# Gradient Boosting does not have built-in class_weight, so we compute sample weights
from sklearn.utils.class_weight import compute_sample_weight
# Compute sample weights to handle class imbalance
sample_weights_inplay = compute_sample_weight('balanced', y_train)
# Fit the model with sample weights
gb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_inplay)

gb_pred = gb_model.predict(X_test_scaled)
gb_pred_proba = gb_model.predict_proba(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)

print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")


if rf_acc >= gb_acc:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_scaler = None
else:
    best_model = gb_model
    best_model_name = "Gradient Boosting"
    best_scaler = scaler

print(f"\n[OK] Best model: {best_model_name} (Accuracy: {max(rf_acc, gb_acc):.4f})")


model_path = MODEL_DIR / "inplay_model.pkl"

# Save the best model along with scaler and feature info
with open(model_path, "wb") as f:
    pickle.dump({
        'model': best_model,
        'model_name': best_model_name,
        'scaler': best_scaler,
        'features': feature_cols,
    }, f)

print(f"Model saved to: {model_path}")

if best_model_name == "Random Forest":
    final_pred = rf_pred
    final_pred_proba = rf_pred_proba
else:
    final_pred = gb_pred
    final_pred_proba = gb_pred_proba

print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_test, final_pred, target_names=['Away Win', 'Draw', 'Home Win']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))

# Calculate weighted ROC-AUC for multi-class if possible
try:
    from sklearn.preprocessing import label_binarize
    # Binarize the output for multi-class ROC-AUC
    y_test_bin = label_binarize(y_test, classes=['A', 'D', 'H'])
    # Calculate ROC-AUC using the predicted probabilities
    roc_auc = roc_auc_score(y_test_bin, final_pred_proba, multi_class='ovr', average='weighted')
    print(f"\nWeighted ROC-AUC: {roc_auc:.4f}")
except:
    print("\nROC-AUC calculation skipped for multi-class")

print("\n" + "="*60)
print("Top 10 Most Important Features:")
print("="*60)

# For tree-based models, we can show feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))

print("\n" + "="*60)
print("Performance by Match Minute:")
print("="*60)

# Analyze accuracy by minute checkpoint
test_df = df.iloc[X_test.index].copy()
test_df['prediction'] = final_pred
test_df['correct'] = test_df['prediction'] == y_test.values

minute_accuracy = test_df.groupby('minute').apply(
    lambda x: accuracy_score(x['result'], x['prediction'])
)

print("\nAccuracy by minute checkpoint:")
for minute, acc in minute_accuracy.items():
    print(f"  {minute:2d}' : {acc:.4f}")



report_path = REPORT_DIR / "inplay_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("IN-PLAY PREDICTION MODEL REPORT\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Model Type: {best_model_name}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"Overall Accuracy: {max(rf_acc, gb_acc):.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, final_pred, target_names=['Away Win', 'Draw', 'Home Win']))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, final_pred)))
    
    f.write("\n\nAccuracy by Minute Checkpoint:\n")
    for minute, acc in minute_accuracy.items():
        f.write(f"  {minute:2d}' : {acc:.4f}\n")
    
    f.write("\n\nTop 10 Features:\n")
    if hasattr(best_model, 'feature_importances_'):
        f.write(feature_importance.head(10).to_string())

print(f"\nReport saved to: {report_path}")

print("\n" + "="*60)
print("[OK] In-play model training complete!")
print("="*60)
