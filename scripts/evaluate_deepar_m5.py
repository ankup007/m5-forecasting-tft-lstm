from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from deepar_m5.data import DataConfig, load_m5_bundle
from deepar_m5.evaluation import (
    compute_holdout_metrics,
    forecast_selected_series,
    load_holdout_actuals,
    write_forecast_csv,
)
from deepar_m5.infer import load_checkpoint
from deepar_m5.model import DeepAR, model_config_from_dict
from deepar_m5.utils import choose_device, configure_logging


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for evaluating a trained checkpoint on holdout data."""

    parser = argparse.ArgumentParser(description="Evaluate a trained DeepAR model on M5 holdout data.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--sales-file", default="sales_train_evaluation.csv", help="Sales file containing the actuals.")
    parser.add_argument("--checkpoint", default="artifacts/deepar_m5/best.pt", help="Path to the trained checkpoint.")
    parser.add_argument("--output-dir", default=None, help="Directory to save evaluation results. Defaults to checkpoint's directory.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--forecast-mode", default="mean", choices=["mean", "sample-mean", "quantile"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load a trained DeepAR model and evaluate its performance on a specified sales file."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Use saved data config but override sales file and data dir
    data_config = DataConfig(**checkpoint["data_config"])
    data_config.data_dir = args.data_dir
    data_config.sales_file = args.sales_file
    
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading bundle for evaluation: %s", args.sales_file)
    bundle = load_m5_bundle(
        data_config,
        encoders=checkpoint["encoders"],
        series_ids=checkpoint["selected_series_ids"],
    )

    model = DeepAR(model_config_from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logger.info("Generating forecasts for %s series", bundle.num_series)
    predictions = forecast_selected_series(
        model,
        bundle,
        data_config,
        args.batch_size,
        device,
        args.forecast_mode,
        args.num_samples,
        args.quantile,
        args.sample_seed,
    )

    selected_ids = bundle.sales_frame["id"].astype(str).tolist()
    logger.info("Loading actuals from %s", args.sales_file)
    actuals = load_holdout_actuals(Path(args.data_dir), args.sales_file, selected_ids, data_config.prediction_length)
    
    logger.info("Computing metrics")
    metrics, series_metrics = compute_holdout_metrics(
        predictions,
        actuals,
        bundle.sales_values,
        bundle,
        Path(args.data_dir),
        data_config.prediction_length,
        compute_wrmsse=True,
    )

    # Save results
    forecasts_path = output_dir / "eval_forecasts.csv"
    metrics_path = output_dir / "eval_metrics.json"
    
    write_forecast_csv(forecasts_path, selected_ids, predictions, actuals, series_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Evaluation complete. Metrics: %s", metrics)
    logger.info("Saved forecasts to %s", forecasts_path)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
