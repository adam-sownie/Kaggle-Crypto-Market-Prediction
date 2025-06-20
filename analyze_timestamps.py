#!/usr/bin/env python3
"""
Script to analyze the timestamp range in train.parquet
to understand the temporal structure of the training data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_timestamps(file_path):
    """Analyze timestamp distribution in the training data."""
    
    print("Loading train.parquet...")
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index name: {df.index.name}")
        print(f"Index dtype: {df.index.dtype}")
        
        # Check the index first (likely where timestamp is)
        print(f"\n{'='*50}")
        print(f"Analyzing INDEX column")
        print(f"{'='*50}")
        
        print(f"Index name: {df.index.name}")
        print(f"Index dtype: {df.index.dtype}")
        print(f"Index length: {len(df.index):,}")
        
        if len(df.index) > 0:
            min_val = df.index.min()
            max_val = df.index.max()
            
            print(f"Min index value: {min_val}")
            print(f"Max index value: {max_val}")
            
            # Try to convert index to datetime if it looks like a timestamp
            try:
                if df.index.dtype in ['int64', 'float64']:
                    # Try different timestamp formats
                    
                    # Unix timestamp in seconds
                    if min_val > 1000000000 and max_val < 4000000000:
                        min_dt = pd.to_datetime(min_val, unit='s')
                        max_dt = pd.to_datetime(max_val, unit='s')
                        print(f"\nAs Unix seconds:")
                        print(f"  Min datetime: {min_dt}")
                        print(f"  Max datetime: {max_dt}")
                        print(f"  Time span: {max_dt - min_dt}")
                    
                    # Unix timestamp in milliseconds
                    elif min_val > 1000000000000:
                        min_dt = pd.to_datetime(min_val, unit='ms')
                        max_dt = pd.to_datetime(max_val, unit='ms')
                        print(f"\nAs Unix milliseconds:")
                        print(f"  Min datetime: {min_dt}")
                        print(f"  Max datetime: {max_dt}")
                        print(f"  Time span: {max_dt - min_dt}")
                    
                    # Unix timestamp in microseconds
                    elif min_val > 1000000000000000:
                        min_dt = pd.to_datetime(min_val, unit='us')
                        max_dt = pd.to_datetime(max_val, unit='us')
                        print(f"\nAs Unix microseconds:")
                        print(f"  Min datetime: {min_dt}")
                        print(f"  Max datetime: {max_dt}")
                        print(f"  Time span: {max_dt - min_dt}")
                    
                    # Unix timestamp in nanoseconds
                    elif min_val > 1000000000000000000:
                        min_dt = pd.to_datetime(min_val, unit='ns')
                        max_dt = pd.to_datetime(max_val, unit='ns')
                        print(f"\nAs Unix nanoseconds:")
                        print(f"  Min datetime: {min_dt}")
                        print(f"  Max datetime: {max_dt}")
                        print(f"  Time span: {max_dt - min_dt}")
                
                else:
                    # Try direct datetime conversion
                    min_dt = pd.to_datetime(min_val)
                    max_dt = pd.to_datetime(max_val)
                    print(f"\nAs datetime:")
                    print(f"  Min datetime: {min_dt}")
                    print(f"  Max datetime: {max_dt}")
                    print(f"  Time span: {max_dt - min_dt}")
            
            except Exception as e:
                print(f"Could not convert index to datetime: {e}")
            
            # Show index distribution
            unique_count = df.index.nunique()
            print(f"\nUnique index values: {unique_count:,}")
            
            # Show sample index values
            print(f"First 10 index values: {list(df.index[:10])}")
            print(f"Last 10 index values: {list(df.index[-10:])}")
            
            # Check if index is sorted
            is_sorted = df.index.is_monotonic_increasing
            print(f"Index is sorted (ascending): {is_sorted}")
        
        # Look for timestamp-related columns in regular columns too
        timestamp_cols = []
        for col in df.columns:
            if any(word in col.lower() for word in ['time', 'date', 'timestamp', 'ts']):
                timestamp_cols.append(col)
        
        print(f"\nPotential timestamp columns in data: {timestamp_cols}")
        
        # If no obvious timestamp columns, look for columns that might be timestamps
        if not timestamp_cols:
            print("\nNo obvious timestamp columns found. Checking for potential timestamp columns...")
            for col in df.columns:
                sample_values = df[col].dropna().head(10)
                print(f"\n{col} sample values: {list(sample_values)}")
                
                # Check if values look like timestamps (large integers or datetime-like)
                if df[col].dtype in ['int64', 'float64']:
                    # Check if could be unix timestamp
                    sample_val = sample_values.iloc[0] if len(sample_values) > 0 else None
                    if sample_val and sample_val > 1000000000:  # Rough unix timestamp range
                        print(f"  -> {col} might be a unix timestamp")
                        timestamp_cols.append(col)
        
        # Analyze each potential timestamp column
        for col in timestamp_cols:
            print(f"\n{'='*50}")
            print(f"Analyzing column: {col}")
            print(f"{'='*50}")
            
            # Basic stats
            print(f"Data type: {df[col].dtype}")
            print(f"Non-null values: {df[col].count():,} / {len(df):,}")
            
            if df[col].count() > 0:
                min_val = df[col].min()
                max_val = df[col].max()
                
                print(f"Min value: {min_val}")
                print(f"Max value: {max_val}")
                
                # Try to convert to datetime if it looks like a timestamp
                try:
                    if df[col].dtype in ['int64', 'float64']:
                        # Try different timestamp formats
                        
                        # Unix timestamp in seconds
                        if min_val > 1000000000 and max_val < 4000000000:
                            min_dt = pd.to_datetime(min_val, unit='s')
                            max_dt = pd.to_datetime(max_val, unit='s')
                            print(f"\nAs Unix seconds:")
                            print(f"  Min datetime: {min_dt}")
                            print(f"  Max datetime: {max_dt}")
                            print(f"  Time span: {max_dt - min_dt}")
                        
                        # Unix timestamp in milliseconds
                        elif min_val > 1000000000000:
                            min_dt = pd.to_datetime(min_val, unit='ms')
                            max_dt = pd.to_datetime(max_val, unit='ms')
                            print(f"\nAs Unix milliseconds:")
                            print(f"  Min datetime: {min_dt}")
                            print(f"  Max datetime: {max_dt}")
                            print(f"  Time span: {max_dt - min_dt}")
                        
                        # Unix timestamp in microseconds
                        elif min_val > 1000000000000000:
                            min_dt = pd.to_datetime(min_val, unit='us')
                            max_dt = pd.to_datetime(max_val, unit='us')
                            print(f"\nAs Unix microseconds:")
                            print(f"  Min datetime: {min_dt}")
                            print(f"  Max datetime: {max_dt}")
                            print(f"  Time span: {max_dt - min_dt}")
                    
                    else:
                        # Try direct datetime conversion
                        min_dt = pd.to_datetime(min_val)
                        max_dt = pd.to_datetime(max_val)
                        print(f"\nAs datetime:")
                        print(f"  Min datetime: {min_dt}")
                        print(f"  Max datetime: {max_dt}")
                        print(f"  Time span: {max_dt - min_dt}")
                
                except Exception as e:
                    print(f"Could not convert to datetime: {e}")
                
                # Show distribution if reasonable number of unique values
                unique_count = df[col].nunique()
                print(f"\nUnique values: {unique_count:,}")
                
                if unique_count < 50:
                    print("Value distribution:")
                    print(df[col].value_counts().head(10))
        
        # If still no timestamp columns found, show first few rows to help identify
        if not timestamp_cols:
            print(f"\n{'='*50}")
            print("No timestamp columns identified. First 5 rows:")
            print(f"{'='*50}")
            print(df.head())
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Make sure you're running this from the correct directory.")
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    # Try different possible paths
    possible_paths = [
        "train.parquet",
        "/kaggle/input/drw-crypto-market-prediction/train.parquet",
        "../train.parquet"
    ]
    
    file_found = False
    for path in possible_paths:
        try:
            analyze_timestamps(path)
            file_found = True
            break
        except FileNotFoundError:
            continue
    
    if not file_found:
        print("Could not find train.parquet in any of the expected locations.")
        print("Please ensure the file exists or modify the path in the script.") 