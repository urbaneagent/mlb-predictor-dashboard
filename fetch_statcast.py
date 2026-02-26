#!/usr/bin/env python3
"""
MLB Statcast Data Fetcher
=========================
Automated script to fetch and store MLB Statcast data.
Uses pybaseball to pull pitch-level data from Baseball Savant.

Usage:
    python fetch_statcast.py              # Fetch yesterday's data
    python fetch_statcast.py --year 2025  # Fetch full 2025 season
    python fetch_statcast.py --range 2025-04-01 2025-09-30  # Custom range

Author: Mike Ross
Date: 2026-02-21
Version: 1.0
"""

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor"

# Year configurations
SEASON_RANGES = {
    2023: ("2023-04-01", "2023-10-01"),
    2024: ("2024-03-28", "2024-10-01"),
    2025: ("2025-03-26", "2025-09-28"),  # 2025 season
    2026: ("2026-03-26", "2026-09-28"),  # 2026 season
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def fetch_date_range(start_date, end_date, verbose=True):
    """Fetch statcast data for a date range"""
    if verbose:
        print(f"📡 Fetching statcast data: {start_date} to {end_date}")
    
    try:
        data = statcast(start_dt=start_date, end_dt=end_date)
        if verbose:
            print(f"   ✅ Got {len(data):,} pitch records")
        return data
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def fetch_season(year, verbose=True):
    """Fetch full season data"""
    if year not in SEASON_RANGES:
        print(f"❌ Unknown season year: {year}")
        print(f"   Available: {list(SEASON_RANGES.keys())}")
        return None
    
    start, end = SEASON_RANGES[year]
    return fetch_date_range(start, end, verbose)


def fetch_yesterday(verbose=True):
    """Fetch yesterday's data"""
    yesterday = datetime.now() - timedelta(days=1)
    return fetch_date_range(yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d"), verbose)


def save_to_parquet(df, year, data_dir=DATA_DIR):
    """Save data to parquet file"""
    if df is None or len(df) == 0:
        print("   ⚠️ No data to save")
        return None
    
    output_path = os.path.join(data_dir, f"statcast_{year}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"   💾 Saved to: {output_path}")
    return output_path


def append_to_combined(df, data_dir=DATA_DIR):
    """Append to combined dataset"""
    combined_path = os.path.join(data_dir, "statcast_2023_2025_RAW.parquet")
    
    if os.path.exists(combined_path):
        existing = pd.read_parquet(combined_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    else:
        combined = df
    
    combined.to_parquet(combined_path, index=False)
    print(f"   💾 Updated combined: {combined_path} ({len(combined):,} total rows)")
    return combined_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MLB Statcast Data Fetcher")
    parser.add_argument("--year", type=int, help="Fetch full season (e.g., 2025)")
    parser.add_argument("--range", nargs=2, metavar=("START", "END"), help="Custom date range")
    parser.add_argument("--yesterday", action="store_true", help="Fetch yesterday's data")
    parser.add_argument("--combine", action="store_true", default=True, help="Append to combined dataset")
    parser.add_argument("--output", help="Output filename (default: statcast_YEAR.parquet)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("⚾ MLB STATCAST DATA FETCHER v1.0")
    print("=" * 60)
    
    df = None
    
    if args.year:
        # Full season
        print(f"\n📅 Fetching {args.year} season...")
        df = fetch_season(args.year)
        if df is not None and args.output:
            save_to_parquet(df, args.output.replace('.parquet', ''), args.output if '/' in args.output else DATA_DIR)
        elif df is not None:
            save_to_parquet(df, args.year)
            
    elif args.range:
        # Custom range
        start, end = args.range
        print(f"\n📅 Fetching custom range: {start} to {end}")
        df = fetch_date_range(start, end)
        
    elif args.yesterday:
        # Yesterday
        print("\n📅 Fetching yesterday's data...")
        df = fetch_yesterday()
        
    else:
        # Default: yesterday
        print("\n📅 Fetching yesterday's data (default)...")
        df = fetch_yesterday()
    
    # Save/append
    if df is not None and len(df) > 0:
        if args.combine:
            append_to_combined(df)
        
        # Also save individual file if year specified
        if args.year:
            save_to_parquet(df, args.year)
    
    print("\n✅ Done!")
    print("\n📊 Usage examples:")
    print("   python fetch_statcast.py --year 2025    # Fetch 2025 season")
    print("   python fetch_statcast.py --yesterday    # Fetch yesterday")
    print("   python fetch_statcast.py --range 2025-07-01 2025-07-31  # July 2025")


if __name__ == "__main__":
    main()
