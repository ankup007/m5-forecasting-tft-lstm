import pandas as pd
import numpy as np
import json
import re
import argparse
from pathlib import Path
import importlib.resources

try:
    from m5_wrmsse import wrmsse
    from m5_wrmsse import data as package_data
except ImportError:
    print("Error: m5_wrmsse package not found. Please ensure it is installed in the current environment.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Calculate WRMSSE using the m5_wrmsse package.")
    parser.add_argument("run_dir", help="Path to the experiment run directory.")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        return

    # Get the standard IDs from the package to ensure correct ordering
    res = importlib.resources.files(package_data)
    with importlib.resources.as_file(res) as resources:
        sales_ids_df = pd.read_csv(resources.joinpath('sales_ids.csv.gz'))
    
    standard_ids = sales_ids_df['id'].tolist()
    
    forecast_files = list(run_dir.glob("holdout_forecasts_*.csv"))
    results = {}
    
    for fpath in forecast_files:
        match = re.search(r"holdout_forecasts_(.*)\.csv", fpath.name)
        if not match:
            continue
            
        mode_full = match.group(1)
        # Handle rounded vs raw
        is_rounded = mode_full.endswith("_rounded")
        mode = mode_full[:-8] if is_rounded else mode_full
        variant = "rounded" if is_rounded else "raw"
        
        print(f"Processing {fpath.name}...")
        try:
            df = pd.read_csv(fpath)
            
            # Map validation IDs to evaluation IDs if necessary to match package data
            df['id'] = df['id'].str.replace('_validation', '_evaluation')
            
            # Reindex to match package order
            df = df.set_index('id').reindex(standard_ids).reset_index()
            
            # Extract F1-F28
            forecast_cols = [f"F{i}" for i in range(1, 29)]
            
            # Check if columns exist
            missing_cols = [c for c in forecast_cols if c not in df.columns]
            if missing_cols:
                print(f"  Warning: Missing columns {missing_cols} in {fpath.name}. Skipping.")
                continue
            
            # Fill NaNs with 0 if any (e.g. if some series were missing in the run)
            if df[forecast_cols].isnull().any().any():
                df[forecast_cols] = df[forecast_cols].fillna(0.0)
                
            forecast_array = df[forecast_cols].values.astype(np.float64)
            
            # Calculate WRMSSE
            score = wrmsse(forecast_array)
            
            if mode not in results:
                results[mode] = {}
            results[mode][variant] = {"wrmsse": float(score)}
            print(f"  {mode} ({variant}): {score:.6f}")
        except Exception as e:
            print(f"  Error processing {fpath.name}: {e}")

    # Save results
    output_path = run_dir / "series_json" / "wrmsse_package.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
