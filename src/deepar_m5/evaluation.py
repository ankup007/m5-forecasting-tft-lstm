from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import DataConfig, WindowSampler, day_number, find_day_columns
from .infer import alternate_submission_id
from .model import DeepAR


logger = logging.getLogger(__name__)


def forecast_selected_series(
    model: DeepAR,
    bundle,
    data_config: DataConfig,
    batch_size: int,
    device: torch.device,
    forecast_mode: str,
    num_samples: int,
    quantile: float,
    sample_seed: int | None,
) -> np.ndarray:
    """Forecast all checkpoint-selected series using mean or sampled summaries."""

    if sample_seed is not None:
        torch.manual_seed(sample_seed)
    sampler = WindowSampler(bundle, data_config.context_length, data_config.prediction_length, seed=data_config.seed)
    predictions = np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32)
    all_indices = np.arange(bundle.num_series)
    for offset in tqdm(range(0, bundle.num_series, batch_size), desc="holdout predict", leave=False):
        series_idx = all_indices[offset : offset + batch_size]
        batch_to_move = sampler.make_inference_batch(series_idx)
        batch = {
            "target": torch.as_tensor(batch_to_move["target"], dtype=torch.float32, device=device),
            "covariates": torch.as_tensor(batch_to_move["covariates"], dtype=torch.float32, device=device),
            "static_cats": torch.as_tensor(batch_to_move["static_cats"], dtype=torch.long, device=device),
            "scale": torch.as_tensor(batch_to_move["scale"], dtype=torch.float32, device=device),
        }
        with torch.no_grad():
            if forecast_mode == "mean":
                pred = model.predict_mean(
                    batch["target"],
                    batch["covariates"],
                    batch["static_cats"],
                    batch["scale"],
                    context_length=data_config.context_length,
                )
            else:
                samples = model.predict_samples(
                    batch["target"],
                    batch["covariates"],
                    batch["static_cats"],
                    batch["scale"],
                    context_length=data_config.context_length,
                    num_samples=num_samples,
                )
                pred = samples.mean(dim=0) if forecast_mode == "sample-mean" else torch.quantile(samples, quantile, dim=0)
        predictions[series_idx] = pred.clamp_min(0.0).cpu().numpy()
    return predictions


def load_holdout_actuals(
    data_dir: Path,
    train_sales_file: str,
    selected_ids: Iterable[str],
    prediction_length: int,
) -> np.ndarray:
    """Load evaluation actuals immediately after the training file's last known day."""

    train_header = pd.read_csv(data_dir / train_sales_file, nrows=0)
    train_day_columns = find_day_columns(train_header.columns)
    holdout_start_day = day_number(train_day_columns[-1]) + 1
    holdout_end_day = holdout_start_day + prediction_length - 1
    evaluation = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    day_columns = find_day_columns(evaluation.columns)
    holdout_columns = [
        column
        for column in day_columns
        if holdout_start_day <= day_number(column) <= holdout_end_day
    ]
    if len(holdout_columns) != prediction_length:
        raise ValueError(
            f"Expected {prediction_length} holdout columns from d_{holdout_start_day} "
            f"to d_{holdout_end_day}, found {len(holdout_columns)}."
        )
    evaluation = evaluation.set_index("id")
    actuals = []
    for series_id in selected_ids:
        eval_id = alternate_submission_id(series_id)
        lookup_id = eval_id if eval_id in evaluation.index else series_id
        actuals.append(evaluation.loc[lookup_id, holdout_columns].to_numpy(dtype=np.float32))
    return np.stack(actuals, axis=0)


def rmsse_denominators(train_values: np.ndarray) -> np.ndarray:
    """Compute per-series RMSSE denominators from the training target history."""

    diffs = np.diff(train_values.astype(np.float64), axis=1)
    denom = np.mean(np.square(diffs), axis=1)
    positive = denom > 0
    if positive.any():
        fallback = float(np.median(denom[positive]))
    else:
        fallback = 1.0
    return np.where(positive, denom, fallback).astype(np.float64)


def bottom_level_revenue_weights(bundle, data_dir: Path, prediction_length: int) -> np.ndarray:
    """Compute bottom-level dollar-sales weights over the last training horizon."""

    day_columns = bundle.day_columns[-prediction_length:]
    sales = bundle.sales_frame[["id", "item_id", "store_id", *day_columns]].copy()
    long_sales = sales.melt(
        id_vars=["id", "item_id", "store_id"],
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
    revenue = long_sales.groupby("id", sort=False)["revenue"].sum().reindex(bundle.sales_frame["id"]).fillna(0.0)
    weights = revenue.to_numpy(dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total


def compute_holdout_metrics(predictions: np.ndarray, actuals: np.ndarray, train_values: np.ndarray, weights: np.ndarray) -> dict:
    """Compute bottom-level holdout metrics for one experiment run."""

    pred = predictions.astype(np.float64)
    actual = actuals.astype(np.float64)
    error = pred - actual
    abs_error = np.abs(error)
    nonzero = actual != 0
    smape_denom = np.abs(actual) + np.abs(pred)
    smape_values = np.zeros_like(abs_error, dtype=np.float64)
    np.divide(2.0 * abs_error, smape_denom, out=smape_values, where=smape_denom > 0)
    rmsse_denom = rmsse_denominators(train_values)
    per_series_rmsse = np.sqrt(np.mean(np.square(error), axis=1) / np.clip(rmsse_denom, 1e-12, None))

    metrics = {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(abs_error.sum() / max(float(np.abs(actual).sum()), 1e-12)),
        "smape": float(np.mean(smape_values)),
        "mape_nonzero": float(np.mean(abs_error[nonzero] / actual[nonzero])) if nonzero.any() else None,
        "rmsse": float(np.mean(per_series_rmsse)),
        "bottom_wrmsse": float(np.sum(weights * per_series_rmsse)),
        "num_series": int(pred.shape[0]),
        "prediction_length": int(pred.shape[1]),
    }
    return metrics

def write_forecast_csv(path: Path, selected_ids: list[str], predictions: np.ndarray, actuals: np.ndarray) -> None:
    """Write selected-series forecasts and actuals for later error analysis."""

    forecast_columns = [f"F{i}" for i in range(1, predictions.shape[1] + 1)]
    actual_columns = [f"actual_F{i}" for i in range(1, actuals.shape[1] + 1)]
    frame = pd.DataFrame({"id": selected_ids})
    frame[forecast_columns] = pd.DataFrame(predictions, index=frame.index)
    frame[actual_columns] = pd.DataFrame(actuals, index=frame.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
