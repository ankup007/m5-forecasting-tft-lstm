from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deepar_m5.data import DataConfig, load_m5_bundle
from deepar_m5.evaluation import compute_holdout_metrics, load_holdout_actuals
from deepar_m5.utils import configure_logging

logger = logging.getLogger(__name__)

def main():
    configure_logging("INFO")
    
    # Use the full dataset for final verification
    config = DataConfig(
        data_dir="m5-forecasting-accuracy",
        sales_file="sales_train_evaluation.csv",
        subset_size=None, 
        prediction_length=28
    )
    
    logger.info("Loading M5 data bundle")
    bundle = load_m5_bundle(config)
    
    # Split into train and holdout actuals
    # The bundle.sales_values contains all known days.
    # We want to forecast the last 28 days using the day before them.
    train_end = bundle.known_days - config.prediction_length
    train_values = bundle.sales_values[:, :train_end]
    
    # Actuals for the holdout period
    selected_ids = bundle.sales_frame["id"].astype(str).tolist()
    actuals = bundle.sales_values[:, train_end:]
    
    logger.info("Generating Naive Forecast (last day observation)")
    # Naive: Repeat the last day of training (day train_end - 1) for the next 28 days
    last_day_observation = train_values[:, -1:]
    predictions_naive = np.repeat(last_day_observation, config.prediction_length, axis=1)
    
    logger.info("Generating Seasonal Naive Forecast (repeat last 28 days)")
    # Seasonal Naive: Repeat the last 28 days of training
    predictions_snaive = train_values[:, -config.prediction_length:]

    for name, predictions in [("Naive", predictions_naive), ("Seasonal Naive", predictions_snaive)]:
        logger.info(f"Computing WRMSSE for {name} Forecast")
        metrics, series_metrics = compute_holdout_metrics(
            predictions,
            actuals,
            train_values,
            bundle,
            Path(config.data_dir),
            config.prediction_length
        )
        
        print(f"\n--- {name} Forecast WRMSSE Verification ---")
        print(f"Subset Size: {config.subset_size}")
        print(f"Total WRMSSE: {metrics['wrmsse']:.5f}")
        print("\nLevel Breakdown (first 5 levels):")
        for i in range(1, 6):
            print(f"Level {i:2d} WRMSSE: {metrics[f'wrmsse_l{i}']:.5f}")
            
        print("\nSample Series Metrics (first 3 rows):")
        print(series_metrics.head(3))
    
    print("\nNote: Author (Paul Morgan) reported ~0.838 (likely Seasonal Naive) for the full dataset.")

if __name__ == "__main__":
    main()
