from __future__ import annotations

import argparse
import itertools
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange

from .data import DataConfig, WindowSampler, config_to_dict, day_number, find_day_columns, load_m5_bundle, save_json
from .infer import alternate_submission_id
from .model import DeepAR, ModelConfig, negative_binomial_nll
from .train import batch_to_torch, choose_device, configure_logging, evaluate, save_checkpoint
from .wandb_utils import add_wandb_args, init_wandb, wandb_finish, wandb_log, wandb_save


logger = logging.getLogger(__name__)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer CLI value."""

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated float CLI value."""

    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    """Parse a comma-separated string CLI value."""

    return [part.strip() for part in value.split(",") if part.strip()]


def run_name(index: int, params: dict[str, int | float | str]) -> str:
    """Build a compact directory name for one experiment configuration."""

    pieces = [
        f"run_{index:03d}",
        f"subset{params['subset_size']}",
        f"ctx{params['context_length']}",
        f"h{params['hidden_size']}",
        f"emb{params['embedding_dim']}",
        f"ep{params['epochs']}",
        f"steps{params['steps_per_epoch']}",
        str(params["forecast_mode"]).replace("-", ""),
    ]
    if params["forecast_mode"] == "quantile":
        pieces.append(f"q{int(float(params['quantile']) * 100):02d}")
    return "_".join(pieces)


def build_training_args(params: dict[str, int | float | str], artifact_dir: Path, base_args: argparse.Namespace) -> SimpleNamespace:
    """Create a Namespace compatible with checkpoint metadata."""

    return SimpleNamespace(
        data_dir=base_args.data_dir,
        sales_file=base_args.train_sales_file,
        artifact_dir=str(artifact_dir),
        subset_size=params["subset_size"],
        context_length=params["context_length"],
        prediction_length=base_args.prediction_length,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        steps_per_epoch=params["steps_per_epoch"],
        hidden_size=params["hidden_size"],
        embedding_dim=params["embedding_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        grad_clip=base_args.grad_clip,
        seed=params["seed"],
        device=base_args.device,
        log_level=base_args.log_level,
        wandb=base_args.wandb,
        wandb_project=base_args.wandb_project,
        wandb_entity=base_args.wandb_entity,
        wandb_run_name=base_args.wandb_run_name,
        wandb_group=base_args.wandb_group,
        wandb_mode=base_args.wandb_mode,
        wandb_tags=base_args.wandb_tags,
    )


def train_one_run(
    params: dict[str, int | float | str],
    artifact_dir: Path,
    base_args: argparse.Namespace,
    device: torch.device,
    wandb_run=None,
) -> tuple[DeepAR, DataConfig, object, list[dict[str, float | int]]]:
    """Train one DeepAR configuration and return the best in-memory model."""

    seed = int(params["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_config = DataConfig(
        data_dir=base_args.data_dir,
        sales_file=base_args.train_sales_file,
        subset_size=int(params["subset_size"]),
        context_length=int(params["context_length"]),
        prediction_length=base_args.prediction_length,
        seed=seed,
    )
    bundle = load_m5_bundle(data_config)
    sampler = WindowSampler(bundle, data_config.context_length, data_config.prediction_length, seed=seed)

    model_config = ModelConfig(
        cardinalities=bundle.cardinalities,
        covariate_dim=len(bundle.covariate_columns),
        hidden_size=int(params["hidden_size"]),
        embedding_dim=int(params["embedding_dim"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
    )
    model = DeepAR(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    train_args = build_training_args(params, artifact_dir, base_args)
    metrics: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())

    save_json(artifact_dir / "encoders.json", bundle.encoders)
    save_json(artifact_dir / "data_config.json", config_to_dict(data_config))
    save_json(artifact_dir / "model_config.json", model.to_config_dict())
    bundle.sales_frame[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].to_csv(
        artifact_dir / "selected_series.csv",
        index=False,
    )

    logger.info("Training %s on %s selected series", artifact_dir.name, bundle.num_series)
    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        progress = trange(int(params["steps_per_epoch"]), desc=f"{artifact_dir.name} epoch {epoch}", leave=False)
        for step_idx in progress:
            global_step = (epoch - 1) * int(params["steps_per_epoch"]) + step_idx + 1
            batch = batch_to_torch(sampler.sample_train_batch(int(params["batch_size"])), device)
            optimizer.zero_grad(set_to_none=True)
            mu, alpha = model(batch["target"], batch["covariates"], batch["static_cats"], batch["scale"])
            loss = negative_binomial_nll(batch["target"], mu, alpha, batch["loss_mask"])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base_args.grad_clip)
            optimizer.step()
            batch_loss = float(loss.item())
            running_loss += batch_loss
            progress.set_postfix(loss=f"{batch_loss:.4f}")
            wandb_log(
                wandb_run,
                {
                    "train/batch_nll": batch_loss,
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": epoch,
                },
                step=global_step,
            )

        train_loss = running_loss / max(int(params["steps_per_epoch"]), 1)
        val_loss = evaluate(model, sampler, int(params["batch_size"]), device)
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("%s epoch=%s train_nll=%.5f val_nll=%.5f", artifact_dir.name, epoch, train_loss, val_loss)
        wandb_log(
            wandb_run,
            {
                "train/epoch_nll": train_loss,
                "validation/nll": val_loss,
                "train/epoch": epoch,
            },
            step=epoch * int(params["steps_per_epoch"]),
        )

        save_checkpoint(
            artifact_dir / "latest.pt",
            model,
            optimizer,
            data_config,
            bundle,
            epoch,
            best_val_loss,
            train_args,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            save_checkpoint(
                artifact_dir / "best.pt",
                model,
                optimizer,
                data_config,
                bundle,
                epoch,
                best_val_loss,
                train_args,
            )
        train_metrics_path = artifact_dir / "train_metrics.json"
        train_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        wandb_save(wandb_run, train_metrics_path)
        wandb_save(wandb_run, artifact_dir / "best.pt")

    model.load_state_dict(best_state)
    model.eval()
    return model, data_config, bundle, metrics


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
        batch = batch_to_torch(sampler.make_inference_batch(series_idx), device)
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


def experiment_grid(args: argparse.Namespace) -> list[dict[str, int | float | str]]:
    """Expand CLI comma-separated values into experiment dictionaries."""

    base_products = itertools.product(
        parse_int_list(args.subset_sizes),
        parse_int_list(args.context_lengths),
        parse_int_list(args.batch_sizes),
        parse_int_list(args.epochs_list),
        parse_int_list(args.steps_per_epoch_list),
        parse_int_list(args.hidden_sizes),
        parse_int_list(args.embedding_dims),
        parse_int_list(args.num_layers_list),
        parse_float_list(args.dropouts),
        parse_float_list(args.learning_rates),
        parse_int_list(args.seeds),
        parse_str_list(args.forecast_modes),
    )
    grid = []
    for values in base_products:
        (
            subset_size,
            context_length,
            batch_size,
            epochs,
            steps_per_epoch,
            hidden_size,
            embedding_dim,
            num_layers,
            dropout,
            learning_rate,
            seed,
            forecast_mode,
        ) = values
        quantiles = parse_float_list(args.quantiles) if forecast_mode == "quantile" else [args.quantile]
        for quantile in quantiles:
            grid.append(
                {
                    "subset_size": subset_size,
                    "context_length": context_length,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "hidden_size": hidden_size,
                    "embedding_dim": embedding_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "forecast_mode": forecast_mode,
                    "quantile": quantile,
                }
            )
    return grid


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for experiment sweeps."""

    parser = argparse.ArgumentParser(description="Run DeepAR M5 validation-to-evaluation experiments.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--train-sales-file", default="sales_train_validation.csv")
    parser.add_argument("--output-dir", default="artifacts/deepar_m5_experiments")
    parser.add_argument("--prediction-length", type=int, default=28)
    parser.add_argument("--subset-sizes", default="100")
    parser.add_argument("--context-lengths", default="56")
    parser.add_argument("--batch-sizes", default="32")
    parser.add_argument("--epochs-list", default="2")
    parser.add_argument("--steps-per-epoch-list", default="20")
    parser.add_argument("--hidden-sizes", default="32")
    parser.add_argument("--embedding-dims", default="8")
    parser.add_argument("--num-layers-list", default="1")
    parser.add_argument("--dropouts", default="0.0")
    parser.add_argument("--learning-rates", default="0.001")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--forecast-modes", default="mean")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--quantiles", default="0.5")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    add_wandb_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run an experiment sweep and write per-run plus summary artifacts."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    if args.prediction_length <= 0:
        raise ValueError("--prediction-length must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = experiment_grid(args)
    summary_rows = []
    logger.info("Running %s experiment configurations", len(grid))
    for run_idx, params in enumerate(grid, start=1):
        artifact_dir = output_dir / run_name(run_idx, params)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "experiment_config.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        wandb_run = init_wandb(
            args,
            config={
                **params,
                "data_dir": args.data_dir,
                "train_sales_file": args.train_sales_file,
                "prediction_length": args.prediction_length,
                "num_samples": args.num_samples,
                "sample_seed": args.sample_seed,
                "device": args.device,
            },
            run_name=args.wandb_run_name or artifact_dir.name,
            group=args.wandb_group or output_dir.name,
        )
        try:
            model, data_config, bundle, train_metrics = train_one_run(params, artifact_dir, args, device, wandb_run)
            predictions = forecast_selected_series(
                model,
                bundle,
                data_config,
                int(params["batch_size"]),
                device,
                str(params["forecast_mode"]),
                args.num_samples,
                float(params["quantile"]),
                args.sample_seed,
            )
            selected_ids = bundle.sales_frame["id"].astype(str).tolist()
            actuals = load_holdout_actuals(Path(args.data_dir), args.train_sales_file, selected_ids, args.prediction_length)
            weights = bottom_level_revenue_weights(bundle, Path(args.data_dir), args.prediction_length)
            holdout_metrics = compute_holdout_metrics(predictions, actuals, bundle.sales_values, weights)

            forecasts_path = artifact_dir / "holdout_forecasts.csv"
            holdout_metrics_path = artifact_dir / "holdout_metrics.json"
            write_forecast_csv(forecasts_path, selected_ids, predictions, actuals)
            holdout_metrics_path.write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")
            wandb_log(wandb_run, {f"holdout/{key}": value for key, value in holdout_metrics.items() if value is not None})
            wandb_save(wandb_run, forecasts_path)
            wandb_save(wandb_run, holdout_metrics_path)

            row = {
                "run": artifact_dir.name,
                **params,
                **holdout_metrics,
                "best_internal_val_nll": min(metric["val_loss"] for metric in train_metrics),
            }
            summary_rows.append(row)
            summary_path = output_dir / "summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            wandb_save(wandb_run, summary_path)
            logger.info("%s holdout metrics: %s", artifact_dir.name, holdout_metrics)
        finally:
            wandb_finish(wandb_run)

    logger.info("Wrote experiment summary to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
