from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Add the project root's src directory to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from deepar_m5.data import DataConfig, load_m5_bundle, save_json
from deepar_m5.evaluation import compute_holdout_metrics, load_holdout_actuals
from deepar_m5.utils import configure_logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate WRMSSE for a DeepAR experiment run.")
    parser.add_argument("run_dir", help="Path to the experiment run directory.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy", help="Path to M5 data.")
    return parser.parse_args()

def main():
    args = parse_args()
    configure_logging("INFO")
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return

    # 1. Load data configuration
    config_path = run_dir / "data_config.json"
    if not config_path.exists():
        logger.error(f"data_config.json not found in {run_dir}")
        return
    
    with open(config_path, "r") as f:
        data_config_dict = json.load(f)
    
    data_config_dict["data_dir"] = args.data_dir
    series_ids_path = run_dir / "selected_series.csv"
    if not series_ids_path.exists():
        logger.error(f"selected_series.csv not found in {run_dir}")
        return
    
    series_df = pd.read_csv(series_ids_path)
    selected_ids = series_df["id"].astype(str).tolist()
    
    data_config = DataConfig(**data_config_dict)
    data_config.subset_size = None
    
    logger.info("Loading M5 data bundle (lightweight)")
    bundle = load_m5_bundle(data_config, series_ids=selected_ids, load_covariates=False)
    
    # 2. Load TRUE holdout actuals (Days 1914-1941)
    logger.info("Loading true holdout actuals from evaluation file")
    actuals = load_holdout_actuals(
        Path(args.data_dir), 
        data_config.sales_file, 
        selected_ids, 
        data_config.prediction_length
    )

    # 3. Find and process forecast CSVs
    forecast_files = list(run_dir.glob("holdout_forecasts_*.csv"))
    wrmsse_results = {}

    for fpath in forecast_files:
        match = re.search(r"holdout_forecasts_(.*)\.csv", fpath.name)
        if not match: continue
        
        mode_full = match.group(1)
        if mode_full.endswith("_rounded"):
            continue # We handle rounding inside the loop
        
        mode = mode_full
        logger.info(f"Processing mode: {mode}")
        
        df = pd.read_csv(fpath)
        df = df.set_index("id").reindex(selected_ids).reset_index()
        forecast_cols = [f"F{i}" for i in range(1, data_config.prediction_length + 1)]
        predictions = df[forecast_cols].to_numpy(dtype=np.float64)
        
        # Calculate WRMSSE using the corrected logic in evaluation.py
        metrics_raw, _ = compute_holdout_metrics(
            predictions,
            actuals,
            bundle.sales_values,
            bundle,
            Path(args.data_dir),
            data_config.prediction_length,
            compute_wrmsse=True,
        )

        rounded_predictions = np.rint(predictions).clip(min=0.0)
        metrics_rounded, _ = compute_holdout_metrics(
            rounded_predictions,
            actuals,
            bundle.sales_values,
            bundle,
            Path(args.data_dir),
            data_config.prediction_length,
            compute_wrmsse=True,
        )
        
        wrmsse_results[mode] = {
            "raw": {
                "wrmsse": metrics_raw["wrmsse"],
                **{k: v for k, v in metrics_raw.items() if k.startswith("wrmsse_l")}
            },
            "rounded": {
                "wrmsse": metrics_rounded["wrmsse"],
                **{k: v for k, v in metrics_rounded.items() if k.startswith("wrmsse_l")}
            }
        }
        logger.info(f"WRMSSE for {mode}: raw={metrics_raw['wrmsse']:.5f}, rounded={metrics_rounded['wrmsse']:.5f}")

    # 4. Save results
    output_path = run_dir / "series_json" / "wrmsse.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(wrmsse_results, f, indent=2)
    logger.info(f"Saved optimized WRMSSE results to {output_path}")

if __name__ == "__main__":
    main()
