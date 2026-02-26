#!/usr/bin/env python3
"""
MLB Predictor v2.1 - Historical Data Pull
Pull 2023-2025 Statcast data + weather for ML training.
"""
import pandas as pd
from pybaseball import statcast
from datetime import datetime
import time

print("="*80)
print("MLB PREDICTOR v2.1 - HISTORICAL DATA PULL")
print("="*80)
print(f"Start time: {datetime.now()}")
print()

# Define seasons
seasons = [
    ('2023', '2023-04-01', '2023-10-31'),
    ('2024', '2024-04-01', '2024-10-31'),
    ('2025', '2025-04-01', '2025-10-31')
]

all_data = []

for season_name, start_date, end_date in seasons:
    print(f"📥 Pulling {season_name} season ({start_date} to {end_date})...")
    start_time = time.time()
    
    try:
        df = statcast(start_date, end_date)
        elapsed = time.time() - start_time
        
        print(f"   ✅ {season_name}: {len(df):,} rows in {elapsed:.1f}s")
        print(f"   📊 Columns: {len(df.columns)}")
        print(f"   💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Add season label
        df['season'] = season_name
        all_data.append(df)
        
        # Rate limit (be nice to MLB servers)
        if season_name != seasons[-1][0]:
            print(f"   ⏳ Waiting 5 seconds before next pull...")
            time.sleep(5)
        
    except Exception as e:
        print(f"   ❌ Error pulling {season_name}: {e}")
        continue

if all_data:
    print()
    print("="*80)
    print("COMBINING SEASONS")
    print("="*80)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Total rows: {len(combined_df):,}")
    print(f"✅ Total columns: {len(combined_df.columns)}")
    print(f"✅ Date range: {combined_df['game_date'].min()} to {combined_df['game_date'].max()}")
    print(f"✅ Unique batters: {combined_df['batter'].nunique():,}")
    print(f"✅ Unique pitchers: {combined_df['pitcher'].nunique():,}")
    print(f"✅ Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Save to parquet (compressed)
    output_file = 'statcast_2023_2025_RAW.parquet'
    print()
    print(f"💾 Saving to {output_file}...")
    combined_df.to_parquet(output_file, compression='snappy', index=False)
    
    file_size_mb = pd.io.common.get_filepath_or_buffer(output_file)[0]
    import os
    file_size_mb = os.path.getsize(output_file) / 1024**2
    
    print(f"✅ Saved! File size: {file_size_mb:.1f} MB")
    
    print()
    print("="*80)
    print("SAMPLE DATA (First Row)")
    print("="*80)
    print(combined_df.iloc[0])
    
    print()
    print(f"End time: {datetime.now()}")
    print("🎉 DATA PULL COMPLETE!")

else:
    print("❌ No data pulled. Check errors above.")

