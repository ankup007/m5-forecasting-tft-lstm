from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import DataConfig, WindowSampler, find_day_columns, load_m5_bundle
from .model import DeepAR, ModelConfig
from .train import batch_to_torch, choose_device, configure_logging


logger = logging.getLogger(__name__)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a PyTorch checkpoint while supporting older PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_fallback_forecasts(data_dir: Path, sales_file: str, prediction_length: int) -> dict[str, np.ndarray]:
    """Build recent-history fallback forecasts for series absent from a pilot model.

    A quick subset model only predicts the item-store ids it was trained on. For
    all other submission rows, this fallback repeats the most recent observed
    horizon from the same sales file used by the checkpoint, so validation-style
    runs do not accidentally use evaluation-period actuals.
    """

    sales_path = data_dir / sales_file
    if not sales_path.exists():
        sales_path = data_dir / "sales_train_validation.csv"
    sales = pd.read_csv(sales_path)
    day_columns = find_day_columns(sales.columns)
    fallback = sales[day_columns[-prediction_length:]].to_numpy(dtype=np.float32)
    forecasts: dict[str, np.ndarray] = {}
    for idx, series_id in enumerate(sales["id"].astype(str)):
        forecasts[series_id] = fallback[idx]
        forecasts[alternate_submission_id(series_id)] = fallback[idx]
    logger.info("Built fallback forecasts from %s for %s ids", sales_path, len(forecasts))
    return forecasts


def normalize_submission_id(series_id: str) -> str:
    """Map M5 validation ids to evaluation ids used by the training file."""

    if series_id.endswith("_validation"):
        return series_id[: -len("_validation")] + "_evaluation"
    return series_id


def alternate_submission_id(series_id: str) -> str:
    """Return the matching M5 id with the other validation/evaluation suffix."""

    if series_id.endswith("_validation"):
        return series_id[: -len("_validation")] + "_evaluation"
    if series_id.endswith("_evaluation"):
        return series_id[: -len("_evaluation")] + "_validation"
    return series_id


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for checkpoint inference and submission writing."""

    parser = argparse.ArgumentParser(description="Run DeepAR inference and write an M5 submission file.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--checkpoint", default="artifacts/deepar_m5/best.pt")
    parser.add_argument("--output", default="artifacts/deepar_m5/submission.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load a trained DeepAR checkpoint and write a complete submission CSV."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    device = choose_device(args.device)
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    data_config = DataConfig(**checkpoint["data_config"])
    data_config.data_dir = args.data_dir

    logger.info("Loading selected series for inference")
    bundle = load_m5_bundle(
        data_config,
        encoders=checkpoint["encoders"],
        series_ids=checkpoint["selected_series_ids"],
    )
    sampler = WindowSampler(bundle, data_config.context_length, data_config.prediction_length, seed=data_config.seed)

    model = DeepAR(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions = np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32)
    all_indices = np.arange(bundle.num_series)
    for offset in tqdm(range(0, bundle.num_series, args.batch_size), desc="predict"):
        series_idx = all_indices[offset : offset + args.batch_size]
        batch_np = sampler.make_inference_batch(series_idx)
        batch = batch_to_torch(batch_np, device)
        with torch.no_grad():
            pred = model.predict_mean(
                batch["target"],
                batch["covariates"],
                batch["static_cats"],
                batch["scale"],
                context_length=data_config.context_length,
            )
        predictions[series_idx] = pred.clamp_min(0.0).cpu().numpy()
    logger.info("Predictions shape=%s", predictions.shape)

    pred_map = {
        series_id: predictions[idx]
        for idx, series_id in enumerate(bundle.sales_frame["id"].astype(str).tolist())
    }
    fallback_map = make_fallback_forecasts(Path(args.data_dir), data_config.sales_file, data_config.prediction_length)

    sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
    forecast_columns = [f"F{i}" for i in range(1, data_config.prediction_length + 1)]
    sample[forecast_columns] = sample[forecast_columns].astype(np.float32)
    output_values = np.zeros((len(sample), data_config.prediction_length), dtype=np.float32)
    model_rows = 0
    fallback_rows = 0
    for row_idx, series_id in enumerate(sample["id"].astype(str)):
        candidate_ids = [series_id, normalize_submission_id(series_id), alternate_submission_id(series_id)]
        for candidate_id in candidate_ids:
            if candidate_id in pred_map:
                output_values[row_idx] = pred_map[candidate_id]
                model_rows += 1
                break
        else:
            for candidate_id in candidate_ids:
                if candidate_id in fallback_map:
                    output_values[row_idx] = fallback_map[candidate_id]
                    fallback_rows += 1
                    break

    sample[forecast_columns] = pd.DataFrame(output_values, columns=forecast_columns, index=sample.index)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    logger.info("Submission rows filled with model=%s fallback=%s total=%s", model_rows, fallback_rows, len(sample))
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
