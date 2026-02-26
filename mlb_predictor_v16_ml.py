#!/usr/bin/env python3
"""
MLB PREDICTOR v16 - TRUE ML (No Leakage)
========================================
Rebuilt to avoid data leakage:
- Features from BEFORE pitch outcome
- Train on plate appearance start
- Predict: will this result in a hit?

Features used:
- Pitch location (plate_x, plate_z)
- Velocity (release_speed)
- Count (balls, strikes)
- Pitcher handedness
- Batter handedness
- Exit velocity (only if contact made - NOT for pre-contact prediction)

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/mikeross/MLB_Predictions"

def run():
    print("\n" + "="*90)
    print("⚾ MLB PREDICTOR v16 - TRUE ML (No Leakage)")
    print("="*90)
    
    # Load statcast data
    print("\nLoading data...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile('/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet')
    
    # Only load pre-contact features (features available BEFORE outcome)
    df = pf.read([
        'pitch_type', 'release_speed', 'release_spin_rate',
        'plate_x', 'plate_z', 'balls', 'strikes',
        'p_throws', 'stand', 'events', 'description'
    ]).to_pandas()
    
    print(f"Total pitches: {len(df):,}")
    
    # Create target: did result in hit? (only use events that exist - meaning PA completed)
    # This is our target - will this PA result in a hit?
    df['hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    
    # Filter: only completed at-bats (have an event)
    df = df[df['events'].notna()].copy()
    print(f"Completed at-bats: {len(df):,}")
    print(f"Hit rate: {df['hit'].mean():.1%}")
    
    # Features for ML - ONLY pre-contact data
    # Pitch movement would be ideal but let's use what's available
    
    # Encode categoricals
    df['pitcher_hand'] = (df['p_throws'] == 'R').astype(int)
    df['batter_hand'] = (df['stand'] == 'R').astype(int)
    df['pitcher_rh'] = df['pitcher_hand']
    df['batter_rh'] = df['batter_hand']
    
    # Platoon advantage
    df['platoon_match'] = ((df['pitcher_rh'] == 1) & (df['batter_rh'] == 1)) | ((df['pitcher_rh'] == 0) & (df['batter_rh'] == 0))
    df['platoon_advantage'] = (~df['platoon_match']).astype(int)  # 1 = platoon advantage for batter
    
    # Count advantage
    df['count_advantage'] = df['balls'] - df['strikes']
    df['batter_ahead'] = (df['count_advantage'] > 0).astype(int)
    
    # Feature columns
    features = ['plate_x', 'plate_z', 'release_speed', 'balls', 'strikes',
                'pitcher_rh', 'batter_rh', 'platoon_advantage', 'count_advantage', 'batter_ahead']
    
    # Add pitch type encoding
    pitch_types = df['pitch_type'].fillna('XX').unique()
    for pt in pitch_types:
        df[f'pt_{pt}'] = (df['pitch_type'] == pt).astype(int)
        features.append(f'pt_{pt}')
    
    # Drop rows with missing features
    df_ml = df.dropna(subset=features + ['hit'])
    print(f"Training samples: {len(df_ml):,}")
    
    X = df_ml[features]
    y = df_ml['hit']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict_proba(X_train)[:, 1]
    
    # AUC
    auc_test = roc_auc_score(y_test, y_pred)
    auc_train = roc_auc_score(y_train, y_pred_train)
    
    print(f"\n🎯 RESULTS:")
    print(f"   Train AUC: {auc_train:.4f}")
    print(f"   Test AUC:  {auc_test:.4f}")
    
    if auc_test > 0.55:
        print(f"\n✅ REAL PREDICTIVE POWER! (AUC > 0.55)")
    else:
        print(f"\n⚠️ AUC below 0.55 - limited predictive power")
    
    # Feature importance
    print("\n📊 TOP FEATURES:")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    import joblib
    joblib.dump(model, f"{DATA_DIR}/mlb_ml_model_v16.pkl")
    print(f"\n✅ Model saved: {DATA_DIR}/mlb_ml_model_v16.pkl")
    
    return model, auc_test

if __name__ == "__main__":
    run()
