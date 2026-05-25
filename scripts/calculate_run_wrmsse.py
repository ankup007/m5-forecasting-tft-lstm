from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from deepar_m5.data import DataConfig, load_m5_bundle, save_json
from deepar_m5.evaluation import compute_holdout_metrics, _aggregate_to_levels, rmsse_denominators
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
    
    # 2. Pre-calculate constant hierarchical data
    train_end = bundle.known_days - data_config.prediction_length
    train_values = bundle.sales_values[:, :train_end].astype(np.float64)
    actuals = bundle.sales_values[:, train_end:].astype(np.float64)
    
    logger.info("Pre-aggregating actuals and training values across 12 levels")
    actual_levels = _aggregate_to_levels(actuals, bundle.sales_frame)
    train_levels = _aggregate_to_levels(train_values, bundle.sales_frame)
    
    logger.info("Computing RMSSE denominators for all levels")
    level_denoms = [rmsse_denominators(t) for t in train_levels]

    logger.info("Computing weights (revenue-based)")
    data_dir = Path(data_config.data_dir)
    day_columns = bundle.day_columns[-data_config.prediction_length:]
    sales = bundle.sales_frame[["item_id", "store_id", *day_columns]].copy()
    long_sales = sales.melt(
        id_vars=["item_id", "store_id"],
        value_vars=day_columns,
        var_name="d",
        value_name="units",
    )
    calendar = pd.read_csv(data_dir / "calendar.csv", usecols=["d", "wm_yr_wk"])
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    long_sales = long_sales.merge(calendar, on="d", how="left")
    long_sales = long_sales.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long_sales["sell_price"] = long_sales["sell_price"].fillna(0.0)
    long_sales["revenue"] = long_sales["units"].astype(float) * long_sales["sell_price"].astype(float)

    bottom_revenue = long_sales.groupby(["item_id", "store_id"], sort=True)["revenue"].sum()
    rev_per_item_store = bottom_revenue.to_dict()
    row_revenue = np.array([rev_per_item_store.get((r.item_id, r.store_id), 0.0) for r in bundle.sales_frame.itertuples()], dtype=np.float64)
    revenue_levels = _aggregate_to_levels(row_revenue[:, None], bundle.sales_frame)
    level_weights = [r.flatten() / np.sum(r).clip(min=1e-12) for r in revenue_levels]

    # 3. Find and process forecast CSVs
    forecast_files = list(run_dir.glob("holdout_forecasts_*.csv"))
    wrmsse_results = {}

    for fpath in forecast_files:
        match = re.search(r"holdout_forecasts_(.*)\.csv", fpath.name)
        if not match: continue
        
        mode_full = match.group(1)
        is_rounded = mode_full.endswith("_rounded")
        mode = mode_full[:-8] if is_rounded else mode_full
        variant = "rounded" if is_rounded else "raw"
        
        logger.info(f"Processing {mode} ({variant})")
        df = pd.read_csv(fpath)
        df = df.set_index("id").reindex(selected_ids).reset_index()
        forecast_cols = [f"F{i}" for i in range(1, data_config.prediction_length + 1)]
        predictions = df[forecast_cols].to_numpy(dtype=np.float64)
        
        # Fast hierarchical aggregation
        pred_levels = _aggregate_to_levels(predictions, bundle.sales_frame)
        
        level_wrmsses = []
        for l_idx in range(12):
            mse = np.mean(np.square(pred_levels[l_idx] - actual_levels[l_idx]), axis=1)
            rmsse = np.sqrt(mse / level_denoms[l_idx])
            level_wrmsses.append(float(np.sum(level_weights[l_idx] * rmsse)))

        wrmsse = float(np.mean(level_wrmsses))
        
        if mode not in wrmsse_results: wrmsse_results[mode] = {}
        # We only save WRMSSE here to keep it lean, or full metrics if needed
        # But for the dashboard we mainly want wrmsse
        wrmsse_results[mode][variant] = {
            "wrmsse": wrmsse,
            **{f"wrmsse_l{i+1}": val for i, val in enumerate(level_wrmsses)}
        }
        logger.info(f"WRMSSE for {mode} {variant}: {wrmsse:.5f}")

    # 4. Save results
    output_path = run_dir / "series_json" / "wrmsse.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(wrmsse_results, f, indent=2)
    logger.info(f"Saved optimized WRMSSE results to {output_path}")

if __name__ == "__main__":
    main()
