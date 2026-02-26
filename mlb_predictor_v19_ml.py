#!/usr/bin/env python3
"""
MLB PREDICTOR v19 - ML FINAL (With Season Stats)
==================================================
Adds pre-game features:
- Batter season stats (AVG, OPS, HR rate)
- Pitcher season stats (ERA, H/IP, HR/IP)

Target: 0.80 AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import joblib

DATA_DIR = "/Users/mikeross/MLB_Predictions"

def run():
    print("\n" + "="*90)
    print("⚾ MLB PREDICTOR v19 - ML FINAL (With Season Stats)")
    print("="*90)
    
    # Load statcast
    print("\nLoading statcast...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile('/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet')
    
    df = pf.read([
        'pitcher', 'batter', 'pitch_type', 'release_speed', 'release_spin_rate',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'plate_x', 'plate_z', 
        'balls', 'strikes',
        'p_throws', 'stand', 'events'
    ]).to_pandas()
    
    # Target
    df['hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    df = df[df['events'].notna()].copy()
    print(f"At-bats: {len(df):,}")
    
    # Load season stats
    print("Loading season stats...")
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    # Create ID mappings
    batter_stats = batters[['Name', 'AVG', 'OPS', 'HR', 'AB']].copy()
    batter_stats['batter_avg'] = batter_stats['AVG']
    batter_stats['batter_ops'] = batter_stats['OPS']
    batter_stats['batter_hr_rate'] = batter_stats['HR'] / batter_stats['AB']
    
    pitcher_stats = pitchers[['Name', 'ERA', 'IP', 'H', 'HR']].copy()
    pitcher_stats['pitcher_era'] = pitcher_stats['ERA']
    pitcher_stats['pitcher_h_ip'] = pitcher_stats['H'] / pitcher_stats['IP']
    pitcher_stats['pitcher_hr_ip'] = pitcher_stats['HR'] / pitcher_stats['IP']
    
    # We need to match by ID - let's use a workaround
    from pybaseball import playerid_reverse_lookup
    
    print("Getting player names...")
    # Get unique pitcher IDs
    pitcher_ids = df['pitcher'].unique()[:100]  # Sample for speed
    batter_ids = df['batter'].unique()[:100]
    
    # Look up names
    pitcher_names = playerid_reverse_lookup(list(pitcher_ids), key_type='mlbam')
    batter_names = playerid_reverse_lookup(list(batter_ids), key_type='mlbam')
    
    pitcher_names['name_lower'] = pitcher_names['name_first'].str.lower() + ' ' + pitcher_names['name_last'].str.lower()
    batter_names['name_lower'] = batter_names['name_first'].str.lower() + ' ' + batter_names['name_last'].str.lower()
    
    # Merge stats
    df = df.merge(pitcher_names[['key_mlbam', 'name_lower']], left_on='pitcher', right_on='key_mlbam', how='left')
    df = df.merge(pitcher_stats, left_on='name_lower', right_on=pitcher_stats['Name'].str.lower(), how='left')
    
    df = df.merge(batter_names[['key_mlbam', 'name_lower']], left_on='batter', right_on='key_mlbam', how='left')
    df = df.merge(batter_stats[['Name', 'batter_avg', 'batter_ops', 'batter_hr_rate']], 
                  left_on='name_lower', right_on=batter_stats['Name'].str.lower(), how='left')
    
    # Fill missing with averages
    df['pitcher_era'] = df['pitcher_era'].fillna(4.00)
    df['pitcher_h_ip'] = df['pitcher_h_ip'].fillna(1.0)
    df['pitcher_hr_ip'] = df['pitcher_hr_ip'].fillna(0.1)
    df['batter_avg'] = df['batter_avg'].fillna(0.250)
    df['batter_ops'] = df['batter_ops'].fillna(0.700)
    df['batter_hr_rate'] = df['batter_hr_rate'].fillna(0.03)
    
    # === FEATURES ===
    df['pitcher_rh'] = (df['p_throws'] == 'R').astype(int)
    df['batter_rh'] = (df['stand'] == 'R').astype(int)
    df['platoon'] = ((df['pitcher_rh'] == 1) & (df['batter_rh'] == 0) | 
                    (df['pitcher_rh'] == 0) & (df['batter_rh'] == 1)).astype(int)
    
    df['count_adv'] = df['balls'] - df['strikes']
    df['two_strikes'] = (df['strikes'] == 2).astype(int)
    df['three_balls'] = (df['balls'] == 3).astype(int)
    
    df['plate_x'] = df['plate_x'].fillna(0)
    df['plate_z'] = df['plate_z'].fillna(2)
    df['heart_zone'] = ((df['plate_x'].abs() < 0.5) & (df['plate_z'].between(1.5, 3.5))).astype(int)
    df['edge_zone'] = ((df['plate_x'].abs() > 0.7) | (df['plate_z'] < 1.0) | (df['plate_z'] > 3.5)).astype(int)
    
    df['speed'] = df['release_speed'].fillna(df['release_speed'].median())
    
    features = [
        'balls', 'strikes', 'count_adv', 'two_strikes', 'three_balls',
        'plate_x', 'plate_z', 'heart_zone', 'edge_zone',
        'speed',
        'pitcher_rh', 'batter_rh', 'platoon',
        'pitcher_era', 'pitcher_h_ip', 'pitcher_hr_ip',
        'batter_avg', 'batter_ops', 'batter_hr_rate'
    ]
    
    df = df.fillna(0)
    df_ml = df.dropna(subset=features + ['hit'])
    print(f"Training samples: {len(df_ml):,}")
    
    X = df_ml[features]
    y = df_ml['hit']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining XGBoost v4...")
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred)
    
    print(f"\n🎯 RESULT: Test AUC = {auc_test:.4f}")
    
    if auc_test >= 0.80:
        print("✅ TARGET 0.80 ACHIEVED!")
    else:
        print(f"⚡ Gap: {0.80 - auc_test:.4f}")
    
    print("\n📊 TOP FEATURES:")
    imp = pd.DataFrame({'f': features, 'i': model.feature_importances_}).sort_values('i', ascending=False)
    for _, r in imp.head(12).iterrows():
        print(f"   {r['f']}: {r['i']:.3f}")
    
    joblib.dump(model, f"{DATA_DIR}/mlb_ml_model_v19.pkl")
    print(f"\n✅ Saved: {DATA_DIR}/mlb_ml_model_v19.pkl")

if __name__ == "__main__":
    run()
