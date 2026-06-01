from __future__ import annotations

import logging
from dataclasses import dataclass
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
_WRMSSE_CONTEXT_CACHE: dict[tuple, tuple[list[tuple[np.ndarray | None, int | None]], dict[int, "WRMSSEContext"]]] = {}


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
            "initial_zero_counter": torch.as_tensor(batch_to_move["initial_zero_counter"], dtype=torch.float32, device=device),
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
                initial_zero_counter=batch["initial_zero_counter"],
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
                initial_zero_counter=batch["initial_zero_counter"],
            )
            samples = samples.clamp_min(0.0)
            if samples.shape != (num_samples, len(series_idx), data_config.prediction_length):
                raise ValueError(f"Unexpected sample forecast shape: {tuple(samples.shape)}")
            
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
            "initial_zero_counter": torch.as_tensor(batch_np["initial_zero_counter"], dtype=torch.float32, device=device),
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
                    initial_zero_counter=batch["initial_zero_counter"],
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
                    initial_zero_counter=batch["initial_zero_counter"],
                )
                if samples.shape != (num_samples, len(series_idx), data_config.prediction_length):
                    raise ValueError(f"Unexpected sample forecast shape: {tuple(samples.shape)}")
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
    evaluation_header = pd.read_csv(data_dir / "sales_train_evaluation.csv", nrows=0)
    day_columns = find_day_columns(evaluation_header.columns)
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
    evaluation = pd.read_csv(data_dir / "sales_train_evaluation.csv", usecols=["id", *holdout_columns])
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


LEVEL_GROUPS = [
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


def _build_level_indices(sales_frame: pd.DataFrame) -> list[tuple[np.ndarray | None, int | None]]:
    """Precompute row-to-aggregate mappings for all M5 hierarchy levels."""

    level_indices: list[tuple[np.ndarray | None, int | None]] = []
    for level_cols in LEVEL_GROUPS:
        if not level_cols or level_cols == ["item_id", "store_id"]:
            level_indices.append((None, None))
            continue
        codes = sales_frame.groupby(level_cols, sort=True).ngroup().to_numpy(dtype=np.int64)
        level_indices.append((codes, int(codes.max()) + 1 if len(codes) else 0))
    return level_indices


def _aggregate_to_levels(
    values: np.ndarray,
    sales_frame: pd.DataFrame,
    level_indices: list[tuple[np.ndarray | None, int | None]] | None = None,
) -> list[np.ndarray]:
    """Aggregate bottom-level series (Level 12) up to all 12 hierarchical levels."""

    if level_indices is None:
        level_indices = _build_level_indices(sales_frame)
    aggregated = []
    for level_idx, level_cols in enumerate(LEVEL_GROUPS):
        if not level_cols:
            # Level 1: Sum everything into a single series
            aggregated.append(values.sum(axis=0, keepdims=True))
        elif level_cols == ["item_id", "store_id"]:
            # Level 12: Already at the bottom level
            aggregated.append(values)
        else:
            codes, group_count = level_indices[level_idx]
            if codes is None or group_count is None:
                raise RuntimeError(f"Missing aggregation index for level {level_idx + 1}")
            agg = np.zeros((group_count, values.shape[1]), dtype=np.float64)
            np.add.at(agg, codes, values)
            aggregated.append(agg)

    return aggregated


@dataclass(frozen=True)
class WRMSSEContext:
    """Reusable M5 WRMSSE inputs for one forecast origin."""

    forecast_start: int
    prediction_length: int
    denominators: list[np.ndarray]
    weights: list[np.ndarray]


def _daily_revenue_matrix(bundle, data_dir: Path, start: int, end: int) -> np.ndarray:
    """Return selected-series daily revenue for the requested observed-day slice."""

    calendar = pd.read_csv(data_dir / "calendar.csv", usecols=["d", "wm_yr_wk"]).set_index("d")
    calendar_weeks = calendar.reindex(bundle.day_columns[start:end])["wm_yr_wk"].to_numpy()
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    weekly_prices = prices.pivot_table(
        index=["store_id", "item_id"],
        columns="wm_yr_wk",
        values="sell_price",
        aggfunc="first",
    )
    series_index = pd.MultiIndex.from_frame(bundle.sales_frame[["store_id", "item_id"]])
    daily_prices = weekly_prices.reindex(series_index).reindex(columns=calendar_weeks).to_numpy(dtype=np.float64)
    daily_prices = np.nan_to_num(daily_prices, nan=0.0)
    return bundle.sales_values[:, start:end] * daily_prices


def precompute_wrmsse_contexts(
    bundle,
    data_dir: Path,
    prediction_length: int,
    forecast_starts: Iterable[int],
) -> tuple[list[tuple[np.ndarray | None, int | None]], dict[int, WRMSSEContext]]:
    """Precompute hierarchy mappings, scales, and revenue weights by forecast origin."""

    origins = sorted({int(origin) for origin in forecast_starts})
    if not origins:
        return _build_level_indices(bundle.sales_frame), {}
    for origin in origins:
        if not 0 < origin <= bundle.known_days:
            raise ValueError(f"WRMSSE forecast origin {origin} must be in [1, {bundle.known_days}].")

    cache_key = (
        str(data_dir.resolve()),
        tuple(bundle.sales_frame["id"].astype(str)),
        tuple(bundle.day_columns),
        int(prediction_length),
        tuple(origins),
    )
    cached = _WRMSSE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        logger.info("Reusing cached WRMSSE contexts for forecast origins: %s", origins)
        return cached

    level_indices = _build_level_indices(bundle.sales_frame)
    sales_levels = _aggregate_to_levels(bundle.sales_values, bundle.sales_frame, level_indices)
    revenue_start = min(max(0, origin - prediction_length) for origin in origins)
    revenue_end = max(origins)
    daily_revenue = _daily_revenue_matrix(bundle, data_dir, revenue_start, revenue_end)
    contexts: dict[int, WRMSSEContext] = {}

    for origin in origins:
        weight_start = max(0, origin - prediction_length)
        bottom_revenue = daily_revenue[:, weight_start - revenue_start : origin - revenue_start].sum(axis=1)
        revenue_levels = _aggregate_to_levels(bottom_revenue[:, None], bundle.sales_frame, level_indices)
        denominators = [rmsse_denominators(values[:, :origin]) for values in sales_levels]
        weights = []
        for revenue in revenue_levels:
            flat_revenue = revenue.reshape(-1)
            weights.append(flat_revenue / max(float(flat_revenue.sum()), 1e-12))
        contexts[origin] = WRMSSEContext(
            forecast_start=origin,
            prediction_length=prediction_length,
            denominators=denominators,
            weights=weights,
        )

    result = (level_indices, contexts)
    _WRMSSE_CONTEXT_CACHE[cache_key] = result
    return result


def compute_wrmsse_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    sales_frame: pd.DataFrame,
    context: WRMSSEContext,
    level_indices: list[tuple[np.ndarray | None, int | None]],
) -> dict[str, float]:
    """Compute competition-style WRMSSE using inputs prepared for one origin."""

    if predictions.shape[1] != context.prediction_length:
        raise ValueError(
            f"Expected prediction length {context.prediction_length}, found {predictions.shape[1]}."
        )
    pred_levels = _aggregate_to_levels(predictions.astype(np.float64), sales_frame, level_indices)
    actual_levels = _aggregate_to_levels(actuals.astype(np.float64), sales_frame, level_indices)
    level_rmsses = []
    level_wrmsses = []

    for pred, actual, denominator, weight in zip(
        pred_levels,
        actual_levels,
        context.denominators,
        context.weights,
    ):
        mse = np.mean(np.square(pred - actual), axis=1)
        rmsse = np.sqrt(mse / denominator)
        level_rmsses.append(float(np.mean(rmsse)))
        level_wrmsses.append(float(np.sum(weight * rmsse)))

    metrics = {
        "wrmsse": float(np.mean(level_wrmsses)),
        "rmsse_l12": level_rmsses[11],
        "wrmsse_l12": level_wrmsses[11],
    }
    for level_idx, value in enumerate(level_wrmsses, start=1):
        metrics[f"wrmsse_l{level_idx}"] = value
    return metrics


def compute_holdout_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    train_values: np.ndarray,
    bundle,
    data_dir: Path,
    prediction_length: int,
    compute_wrmsse: bool = False,
    wrmsse_context: WRMSSEContext | None = None,
    wrmsse_level_indices: list[tuple[np.ndarray | None, int | None]] | None = None,
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

    zero_mask = actual_b <= 0.0
    nonzero_mask = actual_b > 0.0
    positive_pred_mask = pred_b >= 0.5
    train_float = train_values.astype(np.float64)
    nonzero_train = np.where(train_float > 0.0, train_float, np.nan)
    spike_threshold = np.full(train_float.shape[0], 3.0, dtype=np.float64)
    has_nonzero_history = np.any(train_float > 0.0, axis=1)
    if np.any(has_nonzero_history):
        spike_threshold[has_nonzero_history] = np.nanpercentile(
            nonzero_train[has_nonzero_history],
            90,
            axis=1,
        )
    spike_threshold = np.where(np.isfinite(spike_threshold), np.maximum(spike_threshold, 3.0), 3.0)
    spike_threshold_matrix = np.broadcast_to(spike_threshold[:, None], actual_b.shape)
    spike_mask = actual_b >= spike_threshold_matrix

    metrics = {
        "mae": float(abs_error_b.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error_b)))),
        "wape": float(abs_error_b.sum() / max(float(np.abs(actual_b).sum()), 1e-12)),
        "smape": float(np.mean(smape_values_b)),
        "mape": float(np.nanmean(series_mape)) if np.any(~np.isnan(series_mape)) else float("nan"),
        "rmsse": float(np.mean(series_rmsse)),
        "zero_day_mae": float(abs_error_b[zero_mask].mean()) if np.any(zero_mask) else float("nan"),
        "nonzero_day_mae": float(abs_error_b[nonzero_mask].mean()) if np.any(nonzero_mask) else float("nan"),
        "zero_false_positive_rate": float(positive_pred_mask[zero_mask].mean()) if np.any(zero_mask) else float("nan"),
        "nonzero_pred_positive_rate": float(positive_pred_mask[nonzero_mask].mean()) if np.any(nonzero_mask) else float("nan"),
        "spike_day_mae": float(abs_error_b[spike_mask].mean()) if np.any(spike_mask) else float("nan"),
        "spike_hit_rate": float((pred_b[spike_mask] >= spike_threshold_matrix[spike_mask]).mean()) if np.any(spike_mask) else float("nan"),
        "spike_bias": float(error_b[spike_mask].mean()) if np.any(spike_mask) else float("nan"),
        "num_zero_days": int(zero_mask.sum()),
        "num_nonzero_days": int(nonzero_mask.sum()),
        "num_spike_days": int(spike_mask.sum()),
        "num_series": int(pred_b.shape[0]),
        "prediction_length": int(pred_b.shape[1]),
    }

    if compute_wrmsse:
        if wrmsse_context is None or wrmsse_level_indices is None:
            wrmsse_level_indices, contexts = precompute_wrmsse_contexts(
                bundle,
                data_dir,
                prediction_length,
                [bundle.known_days],
            )
            wrmsse_context = contexts[bundle.known_days]
        metrics.update(
            compute_wrmsse_metrics(
                predictions,
                actuals,
                bundle.sales_frame,
                wrmsse_context,
                wrmsse_level_indices,
            )
        )

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
