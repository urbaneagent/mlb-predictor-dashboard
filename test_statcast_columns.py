#!/usr/bin/env python3
"""Quick test: Pull 1 day of Statcast and show all available columns."""
from pybaseball import statcast
import pandas as pd

print("Pulling 1 day of Statcast data (2024-07-01)...")
df = statcast('2024-07-01', '2024-07-01')
print(f"✅ Loaded {len(df)} rows")

print("\n" + "="*80)
print("AVAILABLE COLUMNS IN STATCAST DATA")
print("="*80)
print(f"\nTotal columns: {len(df.columns)}")
print("\nColumn list:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3}. {col}")

print("\n" + "="*80)
print("SAMPLE ROW (first plate appearance)")
print("="*80)
first_row = df.iloc[0]
for col in df.columns:
    val = first_row[col]
    print(f"{col:30} = {val}")
