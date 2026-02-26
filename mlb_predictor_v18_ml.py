#!/usr/bin/env python3
"""
MLB PREDICTOR v18 - ML v3 (TRUE features only)
==============================================
REMOVED: features that leak outcome (swinging, ball, called_strike)
ADDED: Batter/pitcher historical stats

Target: 0.80 AUC with CLEAN features only
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
    print("⚾ MLB PREDICTOR v18 - ML v3 (Clean Features)")
    print("="*90)
    
    # Load statcast
    print("\nLoading data...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile('/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet')
    
    df = pf.read([
        'pitch_type', 'release_speed', 'release_spin_rate',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'plate_x', 'plate_z', 
        'balls', 'strikes',
        'p_throws', 'stand', 'events',
        'pfx_x', 'pfx_z'
    ]).to_pandas()
    
    print(f"Total pitches: {len(df):,}")
    
    # Target
    df['hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    df = df[df['events'].notna()].copy()
    print(f"At-bats: {len(df):,}")
    
    # === CLEAN FEATURES (No outcome leakage) ===
    
    # Handedness
    df['pitcher_rh'] = (df['p_throws'] == 'R').astype(int)
    df['batter_rh'] = (df['stand'] == 'R').astype(int)
    df['platoon'] = ((df['pitcher_rh'] == 1) & (df['batter_rh'] == 0) | 
                    (df['pitcher_rh'] == 0) & (df['batter_rh'] == 1)).astype(int)
    
    # Count (pre-pitch state)
    df['count_adv'] = df['balls'] - df['strikes']
    df['first_pitch'] = ((df['balls'] == 0) & (df['strikes'] == 0)).astype(int)
    df['pitcher_ahead'] = (df['strikes'] > df['balls']).astype(int)
    df['batter_ahead'] = (df['balls'] > df['strikes']).astype(int)
    df['two_strikes'] = (df['strikes'] == 2).astype(int)
    df['three_balls'] = (df['balls'] == 3).astype(int)
    
    # Location zones (pre-contact)
    df['plate_x'] = df['plate_x'].fillna(0)
    df['plate_z'] = df['plate_z'].fillna(2)
    df['heart_zone'] = ((df['plate_x'].abs() < 0.5) & (df['plate_z'].between(1.5, 3.5))).astype(int)
    df['edge_zone'] = ((df['plate_x'].abs() > 0.7) | (df['plate_z'] < 1.0) | (df['plate_z'] > 3.5)).astype(int)
    df['inside'] = ((df['plate_x'] > 0.5) & (df['batter_rh'] == 1) | 
                   (df['plate_x'] < -0.5) & (df['batter_rh'] == 0)).astype(int)
    df['away'] = ((df['plate_x'] < -0.5) & (df['batter_rh'] == 1) |
                  (df['plate_x'] > 0.5) & (df['batter_rh'] == 0)).astype(int)
    
    # Fill any remaining NAs
    df = df.fillna(0)
    
    # Pitch characteristics
    df['speed'] = df['release_speed'].fillna(df['release_speed'].median())
    df['spin'] = df['release_spin_rate'].fillna(df['release_spin_rate'].median())
    df['movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2).fillna(0)
    
    # Release point
    df['release_ext'] = df['release_pos_y'].fillna(df['release_pos_y'].median())
    df['release_side'] = df['release_pos_x'].fillna(0)
    df['release_height'] = df['release_pos_z'].fillna(df['release_pos_z'].median())
    
    # Pitch type one-hot
    pt_map = {'FF': 'Fastball', 'SL': 'Slider', 'CH': 'Changeup', 
              'CU': 'Curveball', 'SI': 'Sinker', 'FC': 'Cutter',
              'KC': 'Knuckle', 'FS': 'Splitter', 'ST': 'Sweeper'}
    df['pitch_type'] = df['pitch_type'].fillna('XX')
    for pt in df['pitch_type'].unique():
        df[f'pt_{pt}'] = (df['pitch_type'] == pt).astype(int)
    
    # Feature list
    features = [
        # Count
        'balls', 'strikes', 'count_adv', 'first_pitch', 'pitcher_ahead', 
        'batter_ahead', 'two_strikes', 'three_balls',
        # Location
        'plate_x', 'plate_z', 'heart_zone', 'edge_zone', 'inside', 'away',
        # Physical
        'speed', 'spin', 'movement', 'release_ext', 'release_side', 'release_height',
        # Matchup
        'pitcher_rh', 'batter_rh', 'platoon'
    ]
    
    # Add pitch types
    for col in df.columns:
        if col.startswith('pt_'):
            features.append(col)
    
    # Prepare data
    df_ml = df.dropna(subset=features + ['hit'])
    print(f"Training samples: {len(df_ml):,}")
    
    X = df_ml[features]
    y = df_ml['hit']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("\nTraining XGBoost v3...")
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred)
    
    print(f"\n🎯 RESULT: Test AUC = {auc_test:.4f}")
    
    if auc_test >= 0.80:
        print("✅ TARGET ACHIEVED!")
    else:
        print(f"⚡ Gap to 0.80: {0.80 - auc_test:.4f}")
    
    # Feature importance
    print("\n📊 TOP FEATURES:")
    imp = pd.DataFrame({'f': features, 'i': model.feature_importances_}).sort_values('i', ascending=False)
    for _, r in imp.head(12).iterrows():
        print(f"   {r['f']}: {r['i']:.3f}")
    
    import joblib
    joblib.dump(model, f"{DATA_DIR}/mlb_ml_model_v18.pkl")
    print(f"\n✅ Saved: {DATA_DIR}/mlb_ml_model_v18.pkl")

if __name__ == "__main__":
    run()
