from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from .data import DataConfig, WindowSampler, config_to_dict, load_m5_bundle, save_json
from .evaluation import (
    compute_holdout_metrics,
    forecast_selected_series,
    load_holdout_actuals,
    write_forecast_csv,
)
from .model import DeepAR, ModelConfig, negative_binomial_nll
from .utils import batch_to_torch, choose_device, configure_logging
from .wandb_utils import add_wandb_args, init_wandb, wandb_finish, wandb_log, wandb_save


logger = logging.getLogger(__name__)


def evaluate(model: DeepAR, sampler: WindowSampler, batch_size: int, device: torch.device) -> float:
    """Evaluate masked validation NLL over deterministic holdout windows."""

    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for batch_np in sampler.iter_validation_batches(batch_size):
            batch = batch_to_torch(batch_np, device)
            mu, alpha = model(
                batch["target"],
                batch["covariates"],
                batch["static_cats"],
                batch["scale"],
                prior_target=batch.get("prior_target"),
            )
            loss_sum = negative_binomial_nll(batch["target"], mu, alpha, batch["loss_mask"])
            weight = float(batch["loss_mask"].sum().item())
            total_loss += float(loss_sum.item()) * weight
            total_weight += weight
    return total_loss / max(total_weight, 1.0)


def save_checkpoint(
    path: Path,
    model: DeepAR,
    optimizer: torch.optim.Optimizer,
    data_config: DataConfig,
    bundle,
    epoch: int,
    best_val_loss: float,
    train_args: argparse.Namespace,
) -> None:
    """Persist model weights plus all metadata needed for reproducible inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.to_config_dict(),
            "data_config": config_to_dict(data_config),
            "encoders": bundle.encoders,
            "covariate_columns": bundle.covariate_columns,
            "selected_series_ids": bundle.sales_frame["id"].astype(str).tolist(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "train_args": vars(train_args),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for DeepAR training runs."""

    parser = argparse.ArgumentParser(description="Train a from-scratch DeepAR model on M5 data.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--sales-file", default="sales_train_evaluation.csv")
    parser.add_argument("--artifact-dir", default="artifacts/deepar_m5")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=56)
    parser.add_argument("--prediction-length", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    # Holdout evaluation arguments
    parser.add_argument("--eval-holdout", action="store_true", help="Run full competition metric evaluation at end.")
    parser.add_argument("--forecast-mode", default="mean", choices=["mean", "sample-mean", "quantile"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--sample-seed", type=int, default=42)
    add_wandb_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Train DeepAR on sampled M5 windows and write checkpoints/artifacts."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    # Sweep support: wandb.init() might have been called by an agent.
    # We call init_wandb which returns existing run if already initialized.
    wandb_run = init_wandb(
        args,
        config={
            **vars(args),
        },
    )

    # If running in a sweep, prioritize wandb.config values
    if wandb_run is not None:
        for key, value in wandb_run.config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        logger.info("Updated arguments from W&B config: %s", wandb_run.config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_config = DataConfig(
        data_dir=args.data_dir,
        sales_file=args.sales_file,
        subset_size=args.subset_size,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        seed=args.seed,
    )
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading M5 data")
    bundle = load_m5_bundle(data_config)
    sampler = WindowSampler(bundle, args.context_length, args.prediction_length, seed=args.seed)
    device = choose_device(args.device)

    model_config = ModelConfig(
        cardinalities=bundle.cardinalities,
        covariate_dim=len(bundle.covariate_columns),
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = DeepAR(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metrics: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_state = None

    # Update W&B with resolved bundle info
    if wandb_run is not None:
        wandb_run.config.update({
            "num_series": bundle.num_series,
            "covariate_dim": len(bundle.covariate_columns),
            "cardinalities": bundle.cardinalities,
            "device_resolved": str(device),
        }, allow_val_change=True)

    save_json(artifact_dir / "encoders.json", bundle.encoders)
    save_json(artifact_dir / "data_config.json", config_to_dict(data_config))
    save_json(artifact_dir / "model_config.json", model.to_config_dict())
    bundle.sales_frame[["id", *["item_id", "dept_id", "cat_id", "store_id", "state_id"]]].to_csv(
        artifact_dir / "selected_series.csv",
        index=False,
    )

    logger.info("Training on %s series, device=%s, covariates=%s", bundle.num_series, device, len(bundle.covariate_columns))
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            progress = trange(args.steps_per_epoch, desc=f"epoch {epoch}", leave=False)
            for step_idx in progress:
                global_step = (epoch - 1) * args.steps_per_epoch + step_idx + 1
                batch_np = sampler.sample_train_batch(args.batch_size)
                batch = batch_to_torch(batch_np, device)
                optimizer.zero_grad(set_to_none=True)
                mu, alpha = model(
                    batch["target"],
                    batch["covariates"],
                    batch["static_cats"],
                    batch["scale"],
                    prior_target=batch.get("prior_target"),
                )
                loss = negative_binomial_nll(batch["target"], mu, alpha, batch["loss_mask"])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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

            train_loss = running_loss / max(args.steps_per_epoch, 1)
            val_loss = evaluate(model, sampler, args.batch_size, device)
            metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            logger.info("epoch=%s train_nll=%.5f val_nll=%.5f", epoch, train_loss, val_loss)
            wandb_log(
                wandb_run,
                {
                    "train/epoch_nll": train_loss,
                    "validation/nll": val_loss,
                    "train/epoch": epoch,
                },
                step=epoch * args.steps_per_epoch,
            )

            save_checkpoint(
                artifact_dir / "latest.pt",
                model,
                optimizer,
                data_config,
                bundle,
                epoch,
                best_val_loss,
                args,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                save_checkpoint(
                    artifact_dir / "best.pt",
                    model,
                    optimizer,
                    data_config,
                    bundle,
                    epoch,
                    best_val_loss,
                    args,
                )

            metrics_path = artifact_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            wandb_save(wandb_run, metrics_path)
            wandb_save(wandb_run, artifact_dir / "best.pt")

        # Optional Holdout Evaluation
        if args.eval_holdout:
            logger.info("Running final holdout evaluation")
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

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
            actuals = load_holdout_actuals(Path(args.data_dir), args.sales_file, selected_ids, args.prediction_length)
            holdout_metrics, series_metrics = compute_holdout_metrics(
                predictions,
                actuals,
                bundle.sales_values,
                bundle,
                Path(args.data_dir),
                args.prediction_length,
            )

            forecasts_path = artifact_dir / "holdout_forecasts.csv"
            holdout_metrics_path = artifact_dir / "holdout_metrics.json"
            write_forecast_csv(forecasts_path, selected_ids, predictions, actuals, series_metrics)
            holdout_metrics_path.write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")

            logger.info("Holdout metrics: %s", holdout_metrics)
            wandb_log(wandb_run, {f"holdout/{k}": v for k, v in holdout_metrics.items() if v is not None})
            wandb_save(wandb_run, forecasts_path)
            wandb_save(wandb_run, holdout_metrics_path)

    finally:
        wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
