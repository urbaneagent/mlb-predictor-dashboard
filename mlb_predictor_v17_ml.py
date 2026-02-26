#!/usr/bin/env python3
"""
MLB PREDICTOR v17 - ML v2 (Target: 0.80 AUC)
==============================================
Enhanced features to push AUC higher:
- More pitch types
- Season stats integration
- Better feature engineering
- Hyperparameter tuning

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
    print("⚾ MLB PREDICTOR v17 - ML v2 (Target: 0.80)")
    print("="*90)
    
    # Load statcast data
    print("\nLoading data...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile('/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet')
    
    # More comprehensive feature set
    df = pf.read([
        'pitch_type', 'release_speed', 'release_spin_rate',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'plate_x', 'plate_z', 
        'balls', 'strikes',
        'p_throws', 'stand', 'events', 'description',
        'hit_speed', 'hit_angle',
        'pfx_x', 'pfx_z'  # Pitch movement
    ]).to_pandas()
    
    print(f"Total pitches: {len(df):,}")
    
    # Target
    df['hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    df = df[df['events'].notna()].copy()
    print(f"Completed at-bats: {len(df):,}")
    
    # === FEATURE ENGINEERING ===
    
    # Basic features
    df['pitcher_rh'] = (df['p_throws'] == 'R').astype(int)
    df['batter_rh'] = (df['stand'] == 'R').astype(int)
    
    # Platoon
    df['platoon_advantage'] = ((df['pitcher_rh'] == 1) & (df['batter_rh'] == 0) | 
                               (df['pitcher_rh'] == 0) & (df['batter_rh'] == 1)).astype(int)
    
    # Count features
    df['count_advantage'] = df['balls'] - df['strikes']
    df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
    df['hitter_count'] = (df['balls'] > df['strikes']).astype(int)
    df['two_strikes'] = (df['strikes'] == 2).astype(int)
    
    # Pitch location zones
    df['zone'] = 0
    df.loc[(df['plate_x'].abs() < 0.5) & (df['plate_z'].between(1.5, 3.5)), 'zone'] = 3  # Heart
    df.loc[(df['plate_x'].abs() < 0.83) & (df['plate_z'].between(1.0, 3.0)), 'zone'] = 2  # Shadow
    df.loc[df['zone'] == 0, 'zone'] = 1  # Edge
    
    # Pitch movement (if available)
    df['movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2).fillna(0)
    df['horizontal_movement'] = df['pfx_x'].fillna(0)
    df['vertical_movement'] = df['pfx_z'].fillna(0)
    
    # Speed features
    df['speed'] = df['release_speed'].fillna(df['release_speed'].median())
    df['speed_squared'] = df['speed'] ** 2
    
    # Location combined
    df['location_combined'] = np.sqrt(df['plate_x']**2 + df['plate_z']**2).fillna(5)
    
    # Pitch type encoding
    df['pitch_type'] = df['pitch_type'].fillna('XX')
    pitch_types = df['pitch_type'].unique()
    for pt in pitch_types:
        df[f'pt_{pt}'] = (df['pitch_type'] == pt).astype(int)
    
    # Description features
    df['swinging'] = (df['description'] == 'swinging_strike').astype(int)
    df['called_strike'] = (df['description'] == 'called_strike').astype(int)
    df['ball'] = (df['description'] == 'ball').astype(int)
    
    # Release position
    df['release_ext'] = df['release_pos_y'].fillna(df['release_pos_y'].median())
    df['release_height'] = df['release_pos_z'].fillna(df['release_pos_z'].median())
    
    # Feature columns
    features = [
        'plate_x', 'plate_z', 'release_speed', 'balls', 'strikes',
        'pitcher_rh', 'batter_rh', 'platoon_advantage',
        'count_advantage', 'full_count', 'hitter_count', 'two_strikes',
        'zone', 'movement', 'horizontal_movement', 'vertical_movement',
        'speed', 'speed_squared', 'location_combined',
        'release_ext', 'release_height',
        'swinging', 'called_strike', 'ball'
    ]
    
    # Add pitch types
    for pt in pitch_types:
        features.append(f'pt_{pt}')
    
    # Clean data
    df_ml = df.dropna(subset=features + ['hit'])
    print(f"Training samples: {len(df_ml):,}")
    
    X = df_ml[features]
    y = df_ml['hit']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with better hyperparameters
    print("\nTraining XGBoost v2...")
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    # Predict
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict_proba(X_train)[:, 1]
    
    # AUC
    auc_test = roc_auc_score(y_test, y_pred)
    auc_train = roc_auc_score(y_train, y_pred_train)
    
    print(f"\n🎯 RESULTS:")
    print(f"   Train AUC: {auc_train:.4f}")
    print(f"   Test AUC:  {auc_test:.4f}")
    
    if auc_test >= 0.80:
        print(f"\n✅ TARGET ACHIEVED! AUC >= 0.80")
    else:
        print(f"\n⚡ Gap to 0.80: {0.80 - auc_test:.4f}")
    
    # Feature importance
    print("\n📊 TOP 15 FEATURES:")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(15).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save
    import joblib
    joblib.dump(model, f"{DATA_DIR}/mlb_ml_model_v17.pkl")
    print(f"\n✅ Model saved: {DATA_DIR}/mlb_ml_model_v17.pkl")
    
    return model, auc_test

if __name__ == "__main__":
    run()
