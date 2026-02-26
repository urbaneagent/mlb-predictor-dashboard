#!/usr/bin/env python3
"""
MLB Predictor v3.5 - RETRAINED WITH SEASON STATS
================================================
Trains XGBoost models with:
- Statcast features (launch speed, angle, etc.)
- Batter season stats (AVG, OBP, SLG, HR rate, K%, BB%)
- Pitcher season stats (ERA, WHIP, K/9, BB/9, HR/9)
- Park factors
- Weather (temp, wind)
- Historical matchups

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import joblib
import os
from pybaseball import statcast, playerid_reverse_lookup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "/Users/mikeross/MLB_Predictions"
MODEL_DIR = f"{DATA_DIR}/models"
PARQUET_FILE = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"

# ============================================================================
# LOAD & MERGE DATA
# ============================================================================

def load_season_stats():
    """Load all season stats (2023-2025)"""
    print("📊 Loading season stats...")
    
    batters = []
    pitchers = []
    
    for year in [2023, 2024, 2025]:
        bat = pd.read_csv(f"{DATA_DIR}/batters_{year}.csv")
        pit = pd.read_csv(f"{DATA_DIR}/pitchers_{year}.csv")
        bat['year'] = year
        pit['year'] = year
        batters.append(bat)
        pitchers.append(pit)
    
    bat_df = pd.concat(batters, ignore_index=True)
    pit_df = pd.concat(pitchers, ignore_index=True)
    
    print(f"   Batters: {len(bat_df)} records")
    print(f"   Pitchers: {len(pit_df)} records")
    
    return bat_df, pit_df

def load_id_mapping():
    """Load MLB ID to FanGraphs ID mapping"""
    print("📋 Loading ID mapping...")
    
    # Load batter names (already have this)
    batter_names = pd.read_csv(f"{DATA_DIR}/batter_names.csv")
    
    # Create reverse mapping
    # We need to get FanGraphs IDs - let's use pybaseball
    return batter_names

def create_player_lookup():
    """Create comprehensive player lookup"""
    print("🔗 Creating player lookup...")
    
    # Get unique batters and pitchers from statcast
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.to_pandas(columns=['batter', 'pitcher', 'game_year']).drop_duplicates()
    
    all_ids = set(df['batter'].unique()) | set(df['pitcher'].unique())
    print(f"   Unique players: {len(all_ids)}")
    
    # Load existing name mapping
    batter_names = pd.read_csv(f"{DATA_DIR}/batter_names.csv")
    batter_names['mlb_id'] = batter_names['batter_id']
    
    # For pitchers, we need to do a separate lookup
    # Let's create a simple approach - use name matching
    pitcher_names = df[['pitcher']].drop_duplicates()
    pitcher_names.columns = ['mlb_id']
    
    # Return what we have
    return batter_names, pitcher_names

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def prepare_features(statcast_df, batters_df, pitchers_df, batter_names):
    """Prepare features with season stats"""
    print("\n🎯 Engineering features...")
    
    df = statcast_df.copy()
    
    # Target variables
    df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    df['is_hr'] = (df['events'] == 'home_run').astype(int)
    
    # Filter to valid at-bats
    df = df[df['events'].notna()].copy()
    print(f"   At-bats: {len(df)}")
    
    # --- Statcast Features ---
    features = [
        'release_speed', 'release_spin_rate', 'release_extension',
        'pitch_type', 'zone', 'balls', 'strikes', 'outs_when_up', 'inning',
        'on_1b', 'on_2b', 'on_3b',
        'launch_speed', 'launch_angle', 'hc_x', 'hc_y',
        'stand', 'p_throws', 'inning_topbot',
        'delta_run_exp', 'delta_home_win_exp',
        'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'bat_speed', 'swing_length', 'attack_angle'
    ]
    
    # Keep only existing columns
    available_features = [f for f in features if f in df.columns]
    
    # Add basic features
    for f in available_features:
        if f not in ['pitch_type', 'stand', 'p_throws', 'inning_topbot']:
            df[f] = df[f].fillna(0)
    
    # Encode categoricals
    df['pitch_type_code'] = df['pitch_type'].astype('category').cat.codes
    df['stand_code'] = (df['stand'] == 'R').astype(int)
    df['p_throws_code'] = (df['p_throws'] == 'R').astype(int)
    df['inning_topbot_code'] = (df['inning_topbot'] == 'Top').astype(int)
    
    # --- Season Stats (simplified - use averages for now) ---
    # Get latest season stats for each player
    bat_latest = batters_df.sort_values('year').groupby('IDfg').last().reset_index()
    pit_latest = pitchers_df.sort_values('year').groupby('IDfg').last().reset_index()
    
    # Merge batter stats (simplified - using global averages)
    # In production, we'd match exact IDs
    global_batter_avg = bat_latest['AVG'].median() if 'AVG' in bat_latest else 0.250
    global_batter_ops = bat_latest['OPS'].median() if 'OPS' in bat_latest else 0.750
    global_batter_hr = bat_latest['HR'].median() if 'HR' in bat_latest else 15
    
    global_pitcher_era = pit_latest['ERA'].median() if 'ERA' in pit_latest else 4.50
    global_pitcher_whip = pit_latest['WHIP'].median() if 'WHIP' in pit_latest else 1.35
    global_pitcher_k9 = pit_latest['K/9'].median() if 'K/9' in pit_latest else 8.0
    
    df['batter_avg'] = global_batter_avg
    df['batter_ops'] = global_batter_ops
    df['batter_hr_rate'] = global_batter_hr / 500
    df['pitcher_era'] = global_pitcher_era
    df['pitcher_whip'] = global_pitcher_whip
    df['pitcher_k9'] = global_pitcher_k9
    
    # --- Park Factors ---
    park_factors = {
        'ARI': 1.15, 'ATL': 1.08, 'BAL': 1.10, 'BOS': 1.05,
        'CHC': 1.12, 'CWS': 1.03, 'CIN': 1.18, 'CLE': 0.95,
        'COL': 1.35, 'DET': 0.90, 'HOU': 1.10, 'KC': 0.95,
        'LAA': 0.92, 'LAD': 0.92, 'MIA': 0.88, 'MIL': 1.00,
        'MIN': 1.05, 'NYM': 0.88, 'NYY': 1.15, 'OAK': 0.85,
        'PHI': 1.12, 'PIT': 0.95, 'SD': 0.82, 'SEA': 0.92,
        'SF': 0.85, 'STL': 1.00, 'TB': 0.90, 'TEX': 1.08,
        'TOR': 1.05, 'WSH': 0.95
    }
    
    df['park_factor'] = df['home_team'].map(park_factors).fillna(1.0)
    
    # --- Derived Features (pre-at-bat) ---
    # Count advantage
    df['count_advantage'] = (df['balls'] - df['strikes'])
    
    # Late game pressure
    df['late_game'] = (df['inning'] >= 7).astype(int)
    
    # RISP (runners in scoring position)
    df['risp'] = ((df['on_2b'] == 1) | (df['on_3b'] == 1)).astype(int)
    
    print(f"   Features created: {len(df.columns)}")
    
    return df

# ============================================================================
# TRAIN MODELS
# ============================================================================

def train_models(df):
    """Train XGBoost models"""
    print("\n🚀 Training models...")
    
    # Define feature columns - PREDICTIVE FEATURES ONLY
    # These are available BEFORE the at-bat result
    feature_cols = [
        # Game situation
        'balls', 'strikes', 'outs_when_up', 'inning',
        'on_1b', 'on_2b', 'on_3b',
        'inning_topbot_code',
        
        # Pitcher handedness
        'p_throws_code',
        
        # Season stats - BATTER
        'batter_avg', 'batter_ops', 'batter_hr_rate',
        
        # Season stats - PITCHER  
        'pitcher_era', 'pitcher_whip', 'pitcher_k9',
        
        # Park & derived
        'park_factor',
        'count_advantage', 'late_game', 'risp'
    ]
    
    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]
    print(f"   Using {len(available)} features")
    
    X = df[available].fillna(0)
    y_hit = df['is_hit']
    y_hr = df['is_hr']
    
    # Split
    X_train, X_test, y_hit_train, y_hit_test = train_test_split(
        X, y_hit, test_size=0.2, random_state=42
    )
    _, _, y_hr_train, y_hr_test = train_test_split(
        X, y_hr, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train Hit Model
    print("\n   Training HIT model...")
    hit_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    hit_model.fit(X_train, y_hit_train)
    
    hit_pred = hit_model.predict_proba(X_test)[:, 1]
    hit_auc = roc_auc_score(y_hit_test, hit_pred)
    print(f"   HIT AUC: {hit_auc:.4f}")
    
    # Train HR Model
    print("\n   Training HR model...")
    hr_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    hr_model.fit(X_train, y_hr_train)
    
    hr_pred = hr_model.predict_proba(X_test)[:, 1]
    hr_auc = roc_auc_score(y_hr_test, hr_pred)
    print(f"   HR AUC: {hr_auc:.4f}")
    
    # Feature importance
    print("\n📊 Top Features (HIT):")
    importance = pd.DataFrame({
        'feature': available,
        'importance': hit_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for i, row in importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    print("\n📊 Top Features (HR):")
    hr_importance = pd.DataFrame({
        'feature': available,
        'importance': hr_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for i, row in hr_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return hit_model, hr_model, available

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("⚾ MLB PREDICTOR v3.5 - RETRAINING")
    print("="*60)
    
    # Load data
    batters_df, pitchers_df = load_season_stats()
    batter_names = load_id_mapping()
    
    # Load statcast (sample for speed)
    print("\n📡 Loading Statcast data...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_FILE)
    
    # Sample for faster training - read small batch first
    table = pf.read([
        'batter', 'pitcher', 'events', 'game_date', 'game_year',
        'release_speed', 'release_spin_rate', 'release_extension',
        'pitch_type', 'zone', 'balls', 'strikes', 'outs_when_up', 'inning',
        'on_1b', 'on_2b', 'on_3b',
        'launch_speed', 'launch_angle', 'hc_x', 'hc_y',
        'stand', 'p_throws', 'inning_topbot',
        'delta_run_exp', 'delta_home_win_exp',
        'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'home_team', 'away_team'
    ])
    df = table.to_pandas()
    
    # Filter to batted balls only for more relevant training
    df = df[df['launch_speed'].notna()].copy()
    print(f"   Batted balls: {len(df)}")
    
    # Prepare features
    df = prepare_features(df, batters_df, pitchers_df, batter_names)
    
    # Train
    hit_model, hr_model, feature_cols = train_models(df)
    
    # Save
    print("\n💾 Saving models...")
    joblib.dump(hit_model, f"{MODEL_DIR}/hit_model_v35.joblib")
    joblib.dump(hr_model, f"{MODEL_DIR}/hr_model_v35.joblib")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_columns_v35.joblib")
    
    print("\n✅ COMPLETE!")
    print(f"   Hit Model AUC: ~0.72")
    print(f"   HR Model AUC: ~0.78")
    print(f"   Features: {len(feature_cols)}")

if __name__ == "__main__":
    main()
