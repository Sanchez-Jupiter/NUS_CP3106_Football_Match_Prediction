"""
Step 2: Pre-match model training
Input: data/processed/pretrain_dataset.csv
Output: 
  - models/pretrain_model.pkl  (Trained model)
  - reports/pretrain_report.txt (Performance evaluation)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = Path("data/processed/pretrain_dataset.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

USE_GPU = True
GPU_DEVICE_ID = "0"


# Load and prepare data
print("Loading pre-match dataset...")
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
    'result', 'goals_home', 'goals_away'
}
candidate_cols = [c for c in df.columns if c not in exclude_cols]
feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

X = df[feature_cols].copy()
y = df['result'].copy()

# Handle missing values
X = X.fillna(0)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFeatures shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")


# Data splitting and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

catboost_device = "GPU" if USE_GPU else "CPU"
xgboost_device = "cuda" if USE_GPU else "cpu"
svm_device = "cpu"

print(f"\nDevice plan -> CatBoost: {catboost_device}, XGBoost: {xgboost_device}, SVM: {svm_device}")

# Standardization
scaler = StandardScaler()# Fit on training data and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train)# Transform test set using the same scaler
X_test_scaled = scaler.transform(X_test)


# Model training: Compare three models
print("\n" + "="*60)
print("Training Model 1: CatBoost")
print("="*60)

'''
class_weights = compute_sample_weight('balanced', y_train)# Calculate class weights for CatBoost
class_weight_values = []
class_counts = np.bincount(y_train, minlength=len(label_encoder.classes_))
for count in class_counts:
    class_weight_values.append(float(len(y_train) / (len(class_counts) * max(count, 1))))
'''
cb_model = CatBoostClassifier(
    iterations=2000,
    depth=10,
    learning_rate=0.05,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=False,
    random_state=42,
    task_type=catboost_device,
    devices=GPU_DEVICE_ID,
    # class_weights=class_weight_values,
)
cb_model.fit(X_train, y_train)

cb_pred = cb_model.predict(X_test).reshape(-1).astype(int)
cb_pred_proba = cb_model.predict_proba(X_test)
cb_acc = accuracy_score(y_test, cb_pred)

print(f"CatBoost Accuracy: {cb_acc:.4f}")

print("\n" + "="*60)
print("Training Model 2: XGBoost")
print("="*60)

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,# Add subsampling to prevent overfitting
    colsample_bytree=0.8,# Add feature subsampling to prevent overfitting
    objective='multi:softprob',# Use softprob for multi-class classification to get probabilities
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    tree_method='hist',
    device=xgboost_device,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"XGBoost Accuracy: {xgb_acc:.4f}")

print("\n" + "="*60)
print("Training Model 3: Support Vector Machine")
print("="*60)

svm_model = SVC(
    kernel='rbf',# Use RBF kernel for non-linear decision boundaries
    C=1.0,# Regularization parameter
    gamma='scale',# Automatically scale gamma based on number of features
    probability=True,
    class_weight='balanced',
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)
svm_pred_proba = svm_model.predict_proba(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"SVM Accuracy: {svm_acc:.4f}")


# Select the best model and save
model_results = {
    'CatBoost': (cb_model, cb_acc, cb_pred, cb_pred_proba, None),
    'XGBoost': (xgb_model, xgb_acc, xgb_pred, xgb_pred_proba, scaler),
    'SVM': (svm_model, svm_acc, svm_pred, svm_pred_proba, scaler),
}

best_model_name, (best_model, best_acc, final_pred, final_pred_proba, best_scaler) = max(
    model_results.items(),
    key=lambda item: item[1][1]
)

print(f"\n[OK] Best model: {best_model_name} (Accuracy: {best_acc:.4f})")

model_path = MODEL_DIR / "pretrain_model.pkl"
scaler_path = MODEL_DIR / "pretrain_scaler.pkl"

# Save the best model and scaler (if applicable)
with open(model_path, "wb") as f:
    pickle.dump({
        'model': best_model,
        'model_name': best_model_name,
        'scaler': best_scaler,
        'features': feature_cols,
        'label_encoder': label_encoder,
        'device_info': {
            'catboost': catboost_device,
            'xgboost': xgboost_device,
            'svm': svm_device,
        },
    }, f)

print(f"Model saved to: {model_path}")

y_test_labels = label_encoder.inverse_transform(y_test)
final_pred_labels = label_encoder.inverse_transform(final_pred.astype(int))

print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_test_labels, final_pred_labels, target_names=['Away Win', 'Draw', 'Home Win']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_labels, final_pred_labels, labels=['A', 'D', 'H']))

# Calculate multi-class ROC-AUC (one-vs-rest)
try:
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test_labels, classes=['A', 'D', 'H'])
    roc_auc = roc_auc_score(y_test_bin, final_pred_proba, multi_class='ovr', average='weighted')
    print(f"\nWeighted ROC-AUC: {roc_auc:.4f}")
except:
    print("\nROC-AUC calculation skipped for multi-class")

# Feature Importance
print("\n" + "="*60)
print("Top 10 Most Important Features:")
print("="*60)

# Only CatBoost and XGBoost provide feature importance directly. SVM does not have a straightforward feature importance measure.
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))

report_path = REPORT_DIR / "pretrain_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("PRE-MATCH PREDICTION MODEL REPORT\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Model Type: {best_model_name}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"CatBoost Device: {catboost_device}\n")
    f.write(f"XGBoost Device: {xgboost_device}\n")
    f.write(f"SVM Device: {svm_device}\n")
    f.write(f"CatBoost Accuracy: {cb_acc:.4f}\n")
    f.write(f"XGBoost Accuracy: {xgb_acc:.4f}\n")
    f.write(f"SVM Accuracy: {svm_acc:.4f}\n")
    f.write(f"Accuracy: {best_acc:.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test_labels, final_pred_labels, target_names=['Away Win', 'Draw', 'Home Win']))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test_labels, final_pred_labels, labels=['A', 'D', 'H'])))
    
    f.write("\n\nTop 10 Features:\n")
    if hasattr(best_model, 'feature_importances_'):
        f.write(feature_importance.head(10).to_string())

print(f"\nReport saved to: {report_path}")

print("\n" + "="*60)
print("[OK] Pre-match model training complete!")
print("="*60)
