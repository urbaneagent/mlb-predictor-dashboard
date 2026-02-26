#!/usr/bin/env python3
"""
MLB Model Training Pipeline v2.1
================================
Trains XGBoost models for Hit and HR prediction
Using 2.1M Statcast plate appearances (2023-2025)

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATH = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"
MODEL_DIR = "/Users/mikeross/MLB_Predictions/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Target events
HIT_EVENTS = ['single', 'double', 'triple', 'home_run']
HR_EVENTS = ['home_run']

# Features to use
FEATURES = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'pitch_type', 'zone', 'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'launch_speed', 'launch_angle', 'hc_x', 'hc_y',
    'stand', 'p_throws', 'inning_topbot',
    'delta_run_exp', 'delta_home_win_exp',
    'n_thruorder_pitcher', 'pitcher_days_since_prev_game',
    'bat_speed', 'swing_length', 'attack_angle'
]

# ============================================================================
# DATA LOADING & PREP
# ============================================================================

print("=" * 60)
print("⚾ MLB MODEL TRAINING PIPELINE v2.1")
print("=" * 60)

print("\n📂 Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"   Loaded {len(df):,} plate appearances")

# Filter to valid batted balls only
print("\n🔧 Preparing features...")

# Create target variables
df['is_hit'] = df['events'].isin(HIT_EVENTS).astype(int)
df['is_hr'] = df['events'].isin(HR_EVENTS).astype(int)

# Filter to only batted balls (where we have launch data)
batted = df[df['launch_speed'].notna()].copy()
print(f"   Batted balls: {len(batted):,}")

# Prepare features
def prepare_features(data):
    """Prepare feature matrix"""
    feat = data[FEATURES].copy()
    
    # Handle categorical
    feat['pitch_type'] = feat['pitch_type'].astype('category').cat.codes
    feat['stand'] = (feat['stand'] == 'R').astype(int)
    feat['p_throws'] = (feat['p_throws'] == 'R').astype(int)
    feat['inning_topbot'] = (feat['inning_topbot'] == 'Top').astype(int)
    
    # Fill NaN
    feat = feat.fillna(0)
    
    return feat

X = prepare_features(batted)
y_hit = batted['is_hit']
y_hr = batted['is_hr']

print(f"   Features: {X.shape[1]}")
print(f"   Hit rate: {y_hit.mean():.1%}")
print(f"   HR rate: {y_hr.mean():.1%}")

# ============================================================================
# TRAIN HIT MODEL
# ============================================================================

print("\n🎯 Training HIT prediction model...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_hit, test_size=0.2, random_state=42, stratify=y_hit
)

# Train XGBoost
hit_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    n_jobs=-1
)

hit_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
y_pred = hit_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"   AUC-ROC: {auc:.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': hit_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Features for HIT prediction:")
for i, row in importance.head(10).iterrows():
    print(f"      {row['feature']}: {row['importance']:.3f}")

# Save model
joblib.dump(hit_model, f"{MODEL_DIR}/hit_model_xgb.joblib")
print(f"\n   ✅ Saved: {MODEL_DIR}/hit_model_xgb.joblib")

# ============================================================================
# TRAIN HR MODEL
# ============================================================================

print("\n🎯 Training HR prediction model...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_hr, test_size=0.2, random_state=42, stratify=y_hr
)

# Train XGBoost (with scale_pos_weight for imbalanced classes)
hr_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=30,  # HR is rare (~3%)
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    n_jobs=-1
)

hr_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
y_pred = hr_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"   AUC-ROC: {auc:.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': hr_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Features for HR prediction:")
for i, row in importance.head(10).iterrows():
    print(f"      {row['feature']}: {row['importance']:.3f}")

# Save model
joblib.dump(hr_model, f"{MODEL_DIR}/hr_model_xgb.joblib")
print(f"\n   ✅ Saved: {MODEL_DIR}/hr_model_xgb.joblib")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModels saved to: {MODEL_DIR}")
print("\nFiles:")
print("   - hit_model_xgb.joblib")
print("   - hr_model_xgb.joblib")

# Create prediction wrapper
wrapper_code = '''
import joblib
import pandas as pd
import numpy as np

class MLBPredictor:
    def __init__(self, model_dir="/Users/mikeross/MLB_Predictions/models"):
        self.hit_model = joblib.load(f"{model_dir}/hit_model_xgb.joblib")
        self.hr_model = joblib.load(f"{model_dir}/hr_model_xgb.joblib")
        self.features = [
            'release_speed', 'release_spin_rate', 'release_extension',
            'pitch_type', 'zone', 'balls', 'strikes', 'outs_when_up', 'inning',
            'on_1b', 'on_2b', 'on_3b',
            'launch_speed', 'launch_angle', 'hc_x', 'hc_y',
            'stand', 'p_throws', 'inning_topbot',
            'delta_run_exp', 'delta_home_win_exp',
            'n_thru_order_pitcher', 'pitcher_days_since_prev_game',
            'bat_speed', 'swing_length', 'attack_angle'
        ]
    
    def predict(self, features_df):
        """Predict hit and HR probability"""
        X = features_df[self.features].fillna(0)
        
        # Handle categorical
        if 'pitch_type' in X.columns:
            X['pitch_type'] = X['pitch_type'].astype('category').cat.codes
        if 'stand' in X.columns:
            X['stand'] = (X['stand'] == 'R').astype(int)
        if 'p_throws' in X.columns:
            X['p_throws'] = (X['p_throws'] == 'R').astype(int)
        if 'inning_topbot' in X.columns:
            X['inning_topbot'] = (X['inning_topbot'] == 'Top').astype(int)
        
        hit_prob = self.hit_model.predict_proba(X)[:, 1]
        hr_prob = self.hr_model.predict_proba(X)[:, 1]
        
        return pd.DataFrame({
            'hit_probability': hit_prob,
            'hr_probability': hr_prob
        })

# Usage:
# predictor = MLBPredictor()
# predictions = predictor.predict(features_df)
'''

with open(f"{MODEL_DIR}/predictor_wrapper.py", 'w') as f:
    f.write(wrapper_code)

print("   - predictor_wrapper.py")

print("\n🚀 Ready for Opening Day March 26, 2026!")
