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


def forecast_multi_summaries(
    model: DeepAR,
    bundle,
    data_config: DataConfig,
    batch_size: int,
    device: torch.device,
    num_samples: int,
    sample_seed: int | None,
) -> dict[str, np.ndarray]:
    """Efficiently generate multiple forecast types (mean, sample-mean, p25, p50, p75) in one go."""

    if sample_seed is not None:
        torch.manual_seed(sample_seed)
    
    sampler = WindowSampler(bundle, data_config.context_length, data_config.prediction_length, seed=data_config.seed)
    
    # Pre-allocate containers for all requested summaries
    results = {
        "mean": np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32),
        "sample-mean": np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32),
        # "p25": np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32),
        # "p50": np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32),
        # "p75": np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32),
    }
    
    all_indices = np.arange(bundle.num_series)
    for offset in tqdm(range(0, bundle.num_series, batch_size), desc="multi-mode predict", leave=False):
        series_idx = all_indices[offset : offset + batch_size]
        batch_to_move = sampler.make_inference_batch(series_idx)
        batch = {
            "target": torch.as_tensor(batch_to_move["target"], dtype=torch.float32, device=device),
            "prior_history": torch.as_tensor(batch_to_move["prior_history"], dtype=torch.float32, device=device),
            "covariates": torch.as_tensor(batch_to_move["covariates"], dtype=torch.float32, device=device),
            "static_cats": torch.as_tensor(batch_to_move["static_cats"], dtype=torch.long, device=device),
            "scale": torch.as_tensor(batch_to_move["scale"], dtype=torch.float32, device=device),
        }
        
        with torch.no_grad():
            # 1. Analytical Mean
            pred_mean = model.predict_mean(
                batch["target"],
                batch["covariates"],
                batch["static_cats"],
                batch["scale"],
                context_length=data_config.context_length,
                prior_history=batch["prior_history"],
            )
            results["mean"][series_idx] = pred_mean.clamp_min(0.0).cpu().numpy()
            
            # 2. Stochastic Samples (do this once)
            samples = model.predict_samples(
                batch["target"],
                batch["covariates"],
                batch["static_cats"],
                batch["scale"],
                context_length=data_config.context_length,
                num_samples=num_samples,
                prior_history=batch["prior_history"],
            )
            samples = samples.clamp_min(0.0)
            
            # 3. Derive stochastic summaries from the same samples
            results["sample-mean"][series_idx] = samples.mean(dim=0).cpu().numpy()
            # results["p25"][series_idx] = torch.quantile(samples, 0.25, dim=0).cpu().numpy()
            # results["p50"][series_idx] = torch.quantile(samples, 0.50, dim=0).cpu().numpy()
            # results["p75"][series_idx] = torch.quantile(samples, 0.75, dim=0).cpu().numpy()
            
    return results


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
    """Generate one selected forecast summary for evaluation scripts."""

    if forecast_mode not in {"mean", "sample-mean", "quantile"}:
        raise ValueError(f"Unknown forecast mode: {forecast_mode}")
    if sample_seed is not None:
        torch.manual_seed(sample_seed)

    sampler = WindowSampler(bundle, data_config.context_length, data_config.prediction_length, seed=data_config.seed)
    predictions = np.zeros((bundle.num_series, data_config.prediction_length), dtype=np.float32)
    all_indices = np.arange(bundle.num_series)

    for offset in tqdm(range(0, bundle.num_series, batch_size), desc="predict", leave=False):
        series_idx = all_indices[offset : offset + batch_size]
        batch_np = sampler.make_inference_batch(series_idx)
        batch = {
            "target": torch.as_tensor(batch_np["target"], dtype=torch.float32, device=device),
            "prior_history": torch.as_tensor(batch_np["prior_history"], dtype=torch.float32, device=device),
            "covariates": torch.as_tensor(batch_np["covariates"], dtype=torch.float32, device=device),
            "static_cats": torch.as_tensor(batch_np["static_cats"], dtype=torch.long, device=device),
            "scale": torch.as_tensor(batch_np["scale"], dtype=torch.float32, device=device),
        }
        with torch.no_grad():
            if forecast_mode == "mean":
                pred = model.predict_mean(
                    batch["target"],
                    batch["covariates"],
                    batch["static_cats"],
                    batch["scale"],
                    context_length=data_config.context_length,
                    prior_history=batch["prior_history"],
                )
            else:
                samples = model.predict_samples(
                    batch["target"],
                    batch["covariates"],
                    batch["static_cats"],
                    batch["scale"],
                    context_length=data_config.context_length,
                    num_samples=num_samples,
                    prior_history=batch["prior_history"],
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
    """Compute per-series RMSSE denominators from the training target history.
    
    In accordance with M5 rules, the denominator is calculated only for the 
    time period after the first non-zero demand.
    """

    denoms = []
    for i in range(train_values.shape[0]):
        series = train_values[i].astype(np.float64)
        # Find first non-zero index
        nonzero_indices = np.nonzero(series)[0]
        if len(nonzero_indices) == 0:
            # Fallback for all-zero series
            diffs = np.diff(series)
        else:
            first_nonzero = nonzero_indices[0]
            # Use only the period after first non-zero demand
            diffs = np.diff(series[first_nonzero:])
        
        if len(diffs) == 0:
            denoms.append(1.0) # Fallback for single-point series
        else:
            denoms.append(np.mean(np.square(diffs)))
            
    return np.clip(np.array(denoms), 1e-12, None)


def _aggregate_to_levels(
    values: np.ndarray,
    sales_frame: pd.DataFrame,
) -> list[np.ndarray]:
    """Aggregate bottom-level series (Level 12) up to all 12 hierarchical levels."""

    levels = [
        [],  # Level 1: Total
        ["state_id"],  # Level 2
        ["store_id"],  # Level 3
        ["cat_id"],  # Level 4
        ["dept_id"],  # Level 5
        ["state_id", "cat_id"],  # Level 6
        ["state_id", "dept_id"],  # Level 7
        ["store_id", "cat_id"],  # Level 8
        ["store_id", "dept_id"],  # Level 9
        ["item_id"],  # Level 10
        ["item_id", "state_id"],  # Level 11
        ["item_id", "store_id"],  # Level 12 (bottom)
    ]

    aggregated = []
    for level_cols in levels:
        if not level_cols:
            # Level 1: Sum everything into a single series
            aggregated.append(values.sum(axis=0, keepdims=True))
        elif level_cols == ["item_id", "store_id"]:
            # Level 12: Already at the bottom level
            aggregated.append(values)
        else:
            # Group and sum based on the specified columns
            temp_df = sales_frame[level_cols].copy()
            # Use a dummy column to ensure we can group and sum the numpy array correctly
            # Actually, it's safer to use pandas grouping directly on the values
            # We convert values to a temporary DataFrame for easy aggregation
            val_df = pd.DataFrame(values)
            combined = pd.concat([temp_df, val_df], axis=1)
            agg_df = combined.groupby(level_cols, sort=True).sum()
            aggregated.append(agg_df.to_numpy(dtype=np.float64))

    return aggregated


def compute_holdout_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    train_values: np.ndarray,
    bundle,
    data_dir: Path,
    prediction_length: int,
    compute_wrmsse: bool = False,
) -> dict:
    """Compute holdout metrics for the M5 competition.

    WRMSSE is computed only when ``compute_wrmsse`` is ``True``.
    """

    # Calculate basic bottom-level metrics (Level 12) per series
    pred_b = predictions.astype(np.float64)
    actual_b = actuals.astype(np.float64)
    error_b = pred_b - actual_b
    abs_error_b = np.abs(error_b)
    
    # Per-series metrics
    series_mae = np.mean(abs_error_b, axis=1)
    series_rmse = np.sqrt(np.mean(np.square(error_b), axis=1))
    
    smape_denom_b = np.abs(actual_b) + np.abs(pred_b)
    smape_values_b = np.zeros_like(abs_error_b, dtype=np.float64)
    np.divide(2.0 * abs_error_b, smape_denom_b, out=smape_values_b, where=smape_denom_b > 0)
    series_smape = np.mean(smape_values_b, axis=1)
    
    # MAPE (only defined where actual > 0)
    mape_values = np.zeros_like(abs_error_b, dtype=np.float64)
    np.divide(abs_error_b, actual_b, out=mape_values, where=actual_b > 0)
    # For series-level MAPE, we can either take mean of available days or handle fully zero actuals
    series_mape = np.array([
        np.mean(mape_values[i][actual_b[i] > 0]) if np.any(actual_b[i] > 0) else np.nan 
        for i in range(len(actual_b))
    ])

    # RMSSE (Level 12)
    denom_b = rmsse_denominators(train_values.astype(np.float64))
    series_rmsse = np.sqrt(np.mean(np.square(error_b), axis=1) / denom_b)

    metrics = {
        "mae": float(abs_error_b.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error_b)))),
        "wape": float(abs_error_b.sum() / max(float(np.abs(actual_b).sum()), 1e-12)),
        "smape": float(np.mean(smape_values_b)),
        "mape": float(np.nanmean(series_mape)) if np.any(~np.isnan(series_mape)) else float("nan"),
        "rmsse": float(np.mean(series_rmsse)),
        "num_series": int(pred_b.shape[0]),
        "prediction_length": int(pred_b.shape[1]),
    }

    if compute_wrmsse:
        # 1. Aggregate unit sales to all 12 levels
        pred_levels = _aggregate_to_levels(predictions.astype(np.float64), bundle.sales_frame)
        actual_levels = _aggregate_to_levels(actuals.astype(np.float64), bundle.sales_frame)
        train_levels = _aggregate_to_levels(train_values.astype(np.float64), bundle.sales_frame)

        # 2. Compute Revenue for each bottom-level series (Level 12)
        day_columns = bundle.day_columns[-prediction_length:]
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

        row_revenue = []
        rev_per_item_store = bottom_revenue.to_dict()
        for row in bundle.sales_frame.itertuples():
            row_revenue.append(rev_per_item_store.get((row.item_id, row.store_id), 0.0))
        row_revenue = np.array(row_revenue, dtype=np.float64)

        # 3. Aggregate revenue to all levels and compute weights
        revenue_levels = _aggregate_to_levels(row_revenue[:, None], bundle.sales_frame)

        level_rmsses = []
        level_wrmsses = []

        for l_idx in range(12):
            p = pred_levels[l_idx]
            a = actual_levels[l_idx]
            t = train_levels[l_idx]
            r = revenue_levels[l_idx].flatten()

            w = r / np.sum(r).clip(min=1e-12)
            mse = np.mean(np.square(p - a), axis=1)
            denom = rmsse_denominators(t)
            rmsse = np.sqrt(mse / denom)

            level_rmsses.append(float(np.mean(rmsse)))
            level_wrmsses.append(float(np.sum(w * rmsse)))

        metrics["wrmsse"] = float(np.mean(level_wrmsses))
        metrics["rmsse_l12"] = level_rmsses[11]
        metrics["wrmsse_l12"] = level_wrmsses[11]
        for i, val in enumerate(level_wrmsses):
            metrics[f"wrmsse_l{i+1}"] = val

    # Per-series breakdown for saving to CSV
    series_metrics = pd.DataFrame({
        "mae": series_mae,
        "rmse": series_rmse,
        "smape": series_smape,
        "mape": series_mape,
        "rmsse": series_rmsse,
    })

    return metrics, series_metrics

def write_forecast_csv(
    path: Path, 
    selected_ids: list[str], 
    predictions: np.ndarray, 
    actuals: np.ndarray,
    series_metrics: pd.DataFrame | None = None,
) -> None:
    """Write selected-series forecasts, actuals, and metrics for error analysis."""

    forecast_columns = [f"F{i}" for i in range(1, predictions.shape[1] + 1)]
    actual_columns = [f"actual_F{i}" for i in range(1, actuals.shape[1] + 1)]
    frame = pd.DataFrame({"id": selected_ids})
    frame[forecast_columns] = pd.DataFrame(predictions, index=frame.index)
    frame[actual_columns] = pd.DataFrame(actuals, index=frame.index)
    
    if series_metrics is not None:
        # Prefix metric columns to distinguish them from forecasts
        metric_cols = series_metrics.copy()
        metric_cols.columns = [f"metric_{c}" for c in metric_cols.columns]
        frame = pd.concat([frame, metric_cols], axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
