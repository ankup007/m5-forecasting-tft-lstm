from __future__ import annotations

import argparse
import itertools
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from .data import DataConfig, WindowSampler, config_to_dict, load_m5_bundle, save_json
from .evaluation import (
    bottom_level_revenue_weights,
    compute_holdout_metrics,
    forecast_selected_series,
    load_holdout_actuals,
    write_forecast_csv,
)
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
