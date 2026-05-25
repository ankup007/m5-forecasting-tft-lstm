from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepar_m5.data import DataConfig, day_number, find_day_columns, load_json, load_m5_bundle


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score saved M5 holdout forecast CSVs without retraining the model."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Experiment run directory containing data_config.json, selected_series.csv, and holdout_forecasts_*.csv.",
    )
    parser.add_argument(
        "--compute-wrmsse",
        action="store_true",
        help="Also compute WRMSSE using the M5 hierarchy and revenue weights.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def list_forecast_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("holdout_forecasts_*.csv"), key=lambda path: path.name)


def load_selected_series_ids(run_dir: Path) -> list[str]:
    selected_path = run_dir / "selected_series.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected_df = pd.read_csv(selected_path, usecols=["id"])
    return selected_df["id"].astype(str).tolist()


def load_predictions(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    forecast_cols = [col for col in frame.columns if col.startswith("F") and col[1:].isdigit()]
    if not forecast_cols:
        raise ValueError(f"No forecast columns found in {path}")
    return frame[forecast_cols].to_numpy(dtype=np.float32)


def alternate_submission_id(series_id: str) -> str:
    if series_id.endswith("_validation"):
        return series_id[: -len("_validation")] + "_evaluation"
    if series_id.endswith("_evaluation"):
        return series_id[: -len("_evaluation")] + "_validation"
    return series_id


def load_holdout_actuals(data_dir: Path, train_sales_file: str, selected_ids: list[str], prediction_length: int) -> np.ndarray:
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
            f"Expected {prediction_length} holdout columns from d_{holdout_start_day} to d_{holdout_end_day}, found {len(holdout_columns)}."
        )
    evaluation = evaluation.set_index("id")
    actuals = []
    for series_id in selected_ids:
        eval_id = alternate_submission_id(series_id)
        lookup_id = eval_id if eval_id in evaluation.index else series_id
        actuals.append(evaluation.loc[lookup_id, holdout_columns].to_numpy(dtype=np.float32))
    return np.stack(actuals, axis=0)


def rmsse_denominators(train_values: np.ndarray) -> np.ndarray:
    denominators = []
    for series in train_values.astype(np.float64):
        nz = np.flatnonzero(series > 0)
        active = series[nz[0] :] if len(nz) else series
        diffs = np.diff(active)
        if diffs.size == 0:
            denominators.append(1e-12)
        else:
            denominators.append(float(np.mean(np.square(diffs))))
    return np.clip(np.asarray(denominators, dtype=np.float64), 1e-12, None)


def aggregate_to_levels(values: np.ndarray, sales_frame: pd.DataFrame) -> list[np.ndarray]:
    levels = [
        [],
        ["state_id"],
        ["store_id"],
        ["cat_id"],
        ["dept_id"],
        ["state_id", "cat_id"],
        ["state_id", "dept_id"],
        ["store_id", "cat_id"],
        ["store_id", "dept_id"],
        ["item_id"],
        ["item_id", "state_id"],
        ["item_id", "store_id"],
    ]

    aggregated: list[np.ndarray] = []
    for level_cols in levels:
        if not level_cols:
            aggregated.append(values.sum(axis=0, keepdims=True))
            continue
        if level_cols == ["item_id", "store_id"]:
            aggregated.append(values)
            continue
        temp_df = sales_frame[level_cols].copy()
        val_df = pd.DataFrame(values)
        combined = pd.concat([temp_df, val_df], axis=1)
        aggregated.append(combined.groupby(level_cols, sort=True).sum().to_numpy(dtype=np.float64))
    return aggregated


def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    train_values: np.ndarray,
    sales_frame: pd.DataFrame,
    data_dir: Path,
    prediction_length: int,
    compute_wrmsse: bool,
) -> dict[str, float]:
    pred_b = predictions.astype(np.float64)
    actual_b = actuals.astype(np.float64)
    error_b = pred_b - actual_b
    abs_error_b = np.abs(error_b)

    smape_denom = np.abs(actual_b) + np.abs(pred_b)
    smape_values = np.zeros_like(abs_error_b, dtype=np.float64)
    np.divide(2.0 * abs_error_b, smape_denom, out=smape_values, where=smape_denom > 0)

    mape_values = np.zeros_like(abs_error_b, dtype=np.float64)
    np.divide(abs_error_b, actual_b, out=mape_values, where=actual_b > 0)
    series_mape = np.array(
        [np.mean(mape_values[i][actual_b[i] > 0]) if np.any(actual_b[i] > 0) else np.nan for i in range(len(actual_b))]
    )

    denom_b = rmsse_denominators(train_values.astype(np.float64))
    series_rmsse = np.sqrt(np.mean(np.square(error_b), axis=1) / denom_b)

    metrics = {
        "mae": float(abs_error_b.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error_b)))),
        "wape": float(abs_error_b.sum() / max(float(np.abs(actual_b).sum()), 1e-12)),
        "smape": float(np.mean(smape_values)),
        "mape": float(np.nanmean(series_mape)) if np.any(~np.isnan(series_mape)) else float("nan"),
        "rmsse": float(np.mean(series_rmsse)),
        "num_series": int(pred_b.shape[0]),
        "prediction_length": int(pred_b.shape[1]),
    }

    if compute_wrmsse:
        pred_levels = aggregate_to_levels(predictions.astype(np.float64), sales_frame)
        actual_levels = aggregate_to_levels(actuals.astype(np.float64), sales_frame)
        train_levels = aggregate_to_levels(train_values.astype(np.float64), sales_frame)

        # Build revenue weights from the last 28 observed training days.
        holdout_cols = find_day_columns(sales_frame.columns)[-prediction_length:]
        sales = sales_frame[["item_id", "store_id", *holdout_cols]].copy()
        long_sales = sales.melt(
            id_vars=["item_id", "store_id"],
            value_vars=holdout_cols,
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
        row_revenue = np.array([bottom_revenue.get((row.item_id, row.store_id), 0.0) for row in sales_frame.itertuples()], dtype=np.float64)
        revenue_levels = aggregate_to_levels(row_revenue[:, None], sales_frame)

        level_wrmsses = []
        for l_idx in range(12):
            p = pred_levels[l_idx]
            a = actual_levels[l_idx]
            t = train_levels[l_idx]
            r = revenue_levels[l_idx].flatten()
            w = r / max(float(np.sum(r)), 1e-12)
            mse = np.mean(np.square(p - a), axis=1)
            denom = rmsse_denominators(t)
            rmsse = np.sqrt(mse / denom)
            level_wrmsses.append(float(np.sum(w * rmsse)))

        metrics["wrmsse"] = float(np.mean(level_wrmsses))
        metrics["rmsse_l12"] = float(np.mean(series_rmsse))
        metrics["wrmsse_l12"] = float(level_wrmsses[11])
        for i, val in enumerate(level_wrmsses):
            metrics[f"wrmsse_l{i+1}"] = float(val)

    return metrics


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    run_dir = Path(args.run_dir)
    data_config_path = run_dir / "data_config.json"
    if not data_config_path.exists():
        raise FileNotFoundError(data_config_path)

    data_config = DataConfig(**load_json(data_config_path))
    selected_ids = load_selected_series_ids(run_dir)

    logger.info("Loading M5 bundle for %s selected series", len(selected_ids))
    bundle = load_m5_bundle(data_config, series_ids=selected_ids)
    actuals = load_holdout_actuals(
        Path(data_config.data_dir),
        data_config.sales_file,
        selected_ids,
        data_config.prediction_length,
    )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    forecast_files = list_forecast_files(run_dir)
    if not forecast_files:
        raise FileNotFoundError(f"No holdout_forecasts_*.csv files found under {run_dir}")

    for forecast_path in forecast_files:
        stem = forecast_path.stem
        if stem.endswith("_rounded"):
            mode = stem[:-len("_rounded")]
            if mode.startswith("holdout_forecasts_"):
                mode = mode[len("holdout_forecasts_"):]
            variant = "rounded"
        else:
            mode = stem[len("holdout_forecasts_"):] if stem.startswith("holdout_forecasts_") else stem
            variant = "raw"

        predictions = load_predictions(forecast_path)
        metrics = compute_metrics(
            predictions=predictions,
            actuals=actuals,
            train_values=bundle.sales_values,
            sales_frame=bundle.sales_frame,
            data_dir=Path(data_config.data_dir),
            prediction_length=data_config.prediction_length,
            compute_wrmsse=args.compute_wrmsse,
        )
        summary.setdefault(mode, {})[variant] = metrics
        logger.info(
            "Scored %s [%s]: MAE=%.5f RMSE=%.5f MAPE=%.5f SMAPE=%.5f RMSSE=%.5f%s",
            mode,
            variant,
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
            metrics["smape"],
            metrics["rmsse"],
            "" if "wrmsse" not in metrics else f" WRMSSE={metrics['wrmsse']:.5f}",
        )

    output_path = run_dir / "scored_holdout_metrics.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote scored metrics to %s", output_path)


if __name__ == "__main__":
    main()
