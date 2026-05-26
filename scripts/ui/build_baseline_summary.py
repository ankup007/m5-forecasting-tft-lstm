from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepar_m5.data import DataConfig, day_number, find_day_columns, load_json, load_m5_bundle


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build naive and seasonal-naive baseline scores for one run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing data_config.json and selected_series.csv.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-series baseline artifacts. Defaults to <run-dir>/../naive_forecasts.",
    )
    parser.add_argument("--output-json", default=None, help="Output file. Defaults to <run-dir>/baseline_summary.json.")
    parser.add_argument("--compare-package", action="store_true", help="Compare WRMSSE against the m5-wrmsse package if installed.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def configure_root_logging(log_level: str) -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")


def alternate_submission_id(series_id: str) -> str:
    if series_id.endswith("_validation"):
        return series_id[: -len("_validation")] + "_evaluation"
    if series_id.endswith("_evaluation"):
        return series_id[: -len("_evaluation")] + "_validation"
    return series_id


def safe_filename(series_id: str) -> str:
    return series_id.replace("/", "__").replace("\\", "__")


def load_holdout_actuals(data_dir: Path, train_sales_file: str, selected_ids: list[str], prediction_length: int) -> np.ndarray:
    train_header = pd.read_csv(data_dir / train_sales_file, nrows=0)
    train_day_columns = find_day_columns(train_header.columns)
    holdout_start_day = day_number(train_day_columns[-1]) + 1
    holdout_end_day = holdout_start_day + prediction_length - 1
    evaluation = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    day_columns = find_day_columns(evaluation.columns)
    holdout_columns = [
        column for column in day_columns if holdout_start_day <= day_number(column) <= holdout_end_day
    ]
    if len(holdout_columns) != prediction_length:
        raise ValueError(
            "Expected %s holdout columns from d_%s to d_%s, found %s."
            % (prediction_length, holdout_start_day, holdout_end_day, len(holdout_columns))
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

        holdout_cols = find_day_columns(sales_frame.columns)[-predictions.shape[1]:]
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
        row_revenue = np.array(
            [bottom_revenue.get((row.item_id, row.store_id), 0.0) for row in sales_frame.itertuples()],
            dtype=np.float64,
        )
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
            metrics["wrmsse_l%s" % (i + 1)] = float(val)

    return metrics


def compute_series_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    train_values: np.ndarray,
) -> dict[str, list[float]]:
    pred_b = predictions.astype(np.float64)
    actual_b = actuals.astype(np.float64)
    error_b = pred_b - actual_b
    abs_error_b = np.abs(error_b)
    smape_denom = np.abs(actual_b) + np.abs(pred_b)
    smape_values = np.zeros_like(abs_error_b, dtype=np.float64)
    np.divide(2.0 * abs_error_b, smape_denom, out=smape_values, where=smape_denom > 0)

    mape_values = np.zeros_like(abs_error_b, dtype=np.float64)
    np.divide(abs_error_b, actual_b, out=mape_values, where=actual_b > 0)

    denom_b = rmsse_denominators(train_values.astype(np.float64))
    series_mse = np.mean(np.square(error_b), axis=1)
    series_rmsse = np.sqrt(series_mse / denom_b)

    series_metrics: list[dict[str, float]] = []
    for i in range(pred_b.shape[0]):
        actual_row = actual_b[i]
        pred_row = pred_b[i]
        abs_row = abs_error_b[i]
        series_metrics.append(
            {
                "mae": float(np.mean(abs_row)),
                "mape": float(np.mean(mape_values[i][actual_row > 0])) if np.any(actual_row > 0) else float("nan"),
                "rmse": float(np.sqrt(series_mse[i])),
                "smape": float(np.mean(smape_values[i])),
                "rmsse": float(series_rmsse[i]),
            }
        )

    return {
        "mae": [float(item["mae"]) for item in series_metrics],
        "mape": [float(item["mape"]) for item in series_metrics],
        "rmse": [float(item["rmse"]) for item in series_metrics],
        "smape": [float(item["smape"]) for item in series_metrics],
        "rmsse": [float(item["rmsse"]) for item in series_metrics],
    }


def package_wrmsse_compare(
    predictions: np.ndarray,
    actuals: np.ndarray,
    bundle,
) -> dict[str, object]:
    try:
        from m5_wrmsse import wrmsse as package_wrmsse
    except Exception as exc:  # pragma: no cover - optional dependency.
        return {"error": str(exc)}

    if bundle.num_series != 30490:
        return {"note": "Skipped package comparison because the selected subset is not the full 30,490-series M5 set."}

    attempts: list[str] = []
    for candidate in (
        lambda: package_wrmsse(predictions),
        lambda: package_wrmsse(predictions, actuals),
    ):
        try:
            value = candidate()
            if isinstance(value, (list, tuple, np.ndarray)):
                value = np.asarray(value).squeeze()
            return {"wrmsse": float(value)}
        except TypeError as exc:
            attempts.append(str(exc))
        except Exception as exc:  # pragma: no cover - optional dependency.
            return {"error": str(exc)}

    return {
        "error": "Unable to call m5_wrmsse.wrmsse with the available signatures.",
        "attempts": attempts,
    }


def metric_payload_from_series(metrics: dict[str, list[float]], index: int) -> dict[str, float]:
    return {name: float(values[index]) for name, values in metrics.items()}


def main() -> None:
    args = build_parser().parse_args()
    configure_root_logging(args.log_level)

    run_dir = Path(args.run_dir)
    data_config_path = run_dir / "data_config.json"
    if not data_config_path.exists():
        raise FileNotFoundError(data_config_path)

    data_config = DataConfig(**load_json(data_config_path))
    full_data_config = DataConfig(
        data_dir=data_config.data_dir,
        sales_file=data_config.sales_file,
        subset_size=None,
        context_length=data_config.context_length,
        prediction_length=data_config.prediction_length,
        seed=data_config.seed,
    )

    logger.info("Loading full M5 series bundle for baseline scoring")
    bundle = load_m5_bundle(full_data_config)
    selected_ids = bundle.sales_frame["id"].astype(str).tolist()
    actuals = load_holdout_actuals(Path(data_config.data_dir), data_config.sales_file, selected_ids, data_config.prediction_length)

    naive = np.repeat(bundle.sales_values[:, -1:], data_config.prediction_length, axis=1)
    seasonal_naive = bundle.sales_values[:, -data_config.prediction_length :]

    naive_metrics = compute_metrics(
        naive,
        actuals,
        bundle.sales_values,
        bundle.sales_frame,
        Path(data_config.data_dir),
        compute_wrmsse=True,
    )
    seasonal_naive_metrics = compute_metrics(
        seasonal_naive,
        actuals,
        bundle.sales_values,
        bundle.sales_frame,
        Path(data_config.data_dir),
        compute_wrmsse=True,
    )

    naive_series_metrics = compute_series_metrics(naive, actuals, bundle.sales_values)
    seasonal_series_metrics = compute_series_metrics(seasonal_naive, actuals, bundle.sales_values)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir.parent / "naive_forecasts"
    series_dir = output_dir / "series"
    output_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    series_ids = selected_ids
    horizon = list(range(1, data_config.prediction_length + 1))
    index_payload = {
        "generated_at": datetime.now().isoformat(),
        "source_run": run_dir.name,
        "source_run_dir": str(run_dir),
        "series_count": len(series_ids),
        "series_ids": series_ids,
        "available_baselines": ["naive", "seasonal_naive"],
    }
    (output_dir / "series_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "source_run": run_dir.name,
        "source_run_dir": str(run_dir),
        "series_count": len(series_ids),
        "available_baselines": ["naive", "seasonal_naive"],
        "baselines": {
            "naive": naive_metrics,
            "seasonal_naive": seasonal_naive_metrics,
        },
    }

    if args.compare_package:
        naive_package = package_wrmsse_compare(naive, actuals, bundle)
        seasonal_package = package_wrmsse_compare(seasonal_naive, actuals, bundle)
        package_compare: dict[str, object] = {}
        if "wrmsse" in naive_package:
            package_compare["naive_wrmsse"] = naive_package["wrmsse"]
        if "wrmsse" in seasonal_package:
            package_compare["seasonal_naive_wrmsse"] = seasonal_package["wrmsse"]

        notes: list[str] = []
        for payload in (naive_package, seasonal_package):
            if "note" in payload:
                notes.append(str(payload["note"]))
            if "error" in payload:
                notes.append(str(payload["error"]))
        if notes:
            package_compare["note"] = "; ".join(dict.fromkeys(notes))
        summary["package_compare"] = package_compare

    summary["naive"] = naive_metrics
    summary["seasonal_naive"] = seasonal_naive_metrics

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Backward-compatible alias for older tooling.
    (run_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for idx, series_id in enumerate(series_ids):
        series_payload = {
            "series_id": series_id,
            "source_run": run_dir.name,
            "horizon": horizon,
            "actuals": [float(v) for v in actuals[idx].tolist()],
            "baselines": {
                "naive": {
                    "forecast": [float(v) for v in naive[idx].tolist()],
                    "metrics": metric_payload_from_series(naive_series_metrics, idx),
                },
                "seasonal_naive": {
                    "forecast": [float(v) for v in seasonal_naive[idx].tolist()],
                    "metrics": metric_payload_from_series(seasonal_series_metrics, idx),
                },
            },
        }
        (series_dir / f"{safe_filename(series_id)}.json").write_text(
            json.dumps(series_payload, indent=2),
            encoding="utf-8",
        )

    output_json = Path(args.output_json) if args.output_json else run_dir / "baseline_summary.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote baseline summary to %s", output_json)


if __name__ == "__main__":
    main()
