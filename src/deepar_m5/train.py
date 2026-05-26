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
    forecast_multi_summaries,
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
                prior_history=batch.get("prior_history"),
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


def train_epoch(
    model: DeepAR,
    sampler: WindowSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    steps_per_epoch: int,
    batch_size: int,
    grad_clip: float,
    wandb_run=None,
) -> float:
    """Run teacher-forced training for a single epoch and return the mean NLL."""

    model.train()
    running_loss = 0.0
    progress = trange(steps_per_epoch, desc=f"epoch {epoch}", leave=False)
    for step_idx in progress:
        global_step = (epoch - 1) * steps_per_epoch + step_idx + 1
        batch_np = sampler.sample_train_batch(batch_size)
        batch = batch_to_torch(batch_np, device)
        optimizer.zero_grad(set_to_none=True)
        mu, alpha = model(
            batch["target"],
            batch["covariates"],
            batch["static_cats"],
            batch["scale"],
            prior_history=batch.get("prior_history"),
        )
        loss = negative_binomial_nll(batch["target"], mu, alpha, batch["loss_mask"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    return running_loss / max(steps_per_epoch, 1)


def save_artifacts(
    artifact_dir: Path,
    bundle,
    data_config: DataConfig,
    model: DeepAR,
    args: argparse.Namespace,
) -> None:
    """Save configuration and metadata files needed for later inference."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_json(artifact_dir / "encoders.json", bundle.encoders)
    save_json(artifact_dir / "data_config.json", config_to_dict(data_config))
    save_json(artifact_dir / "model_config.json", model.to_config_dict())
    
    # Save a unified run_config.json with key hyperparameters for easy reference
    run_config = {
        "subset_size": getattr(args, "subset_size", data_config.subset_size),
        "context_length": getattr(args, "context_length", data_config.context_length),
        "prediction_length": getattr(args, "prediction_length", data_config.prediction_length),
        "batch_size": getattr(args, "batch_size", 128),
        "epochs": getattr(args, "epochs", 10),
        "steps_per_epoch": getattr(args, "steps_per_epoch", 200),
        "hidden_size": getattr(args, "hidden_size", 64),
        "embedding_dim": getattr(args, "embedding_dim", 16),
        "num_layers": getattr(args, "num_layers", 1),
        "dropout": getattr(args, "dropout", 0.0),
        "learning_rate": getattr(args, "learning_rate", 1e-3),
        "grad_clip": getattr(args, "grad_clip", 10.0),
        "seed": getattr(args, "seed", 42),
    }
    save_json(artifact_dir / "run_config.json", run_config)
    
    bundle.sales_frame[["id", *["item_id", "dept_id", "cat_id", "store_id", "state_id"]]].to_csv(
        artifact_dir / "selected_series.csv",
        index=False,
    )


def run_holdout_evaluation(
    model: DeepAR,
    bundle,
    data_config: DataConfig,
    artifact_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    wandb_run=None,
    compute_wrmsse: bool = False,
) -> None:
    """Run the 12-level competitive evaluation for multiple forecast summaries."""

    logger.info("Running multi-mode holdout evaluation")
    model.eval()

    # Generate all requested summaries in one efficient pass
    forecasts_dict = forecast_multi_summaries(
        model,
        bundle,
        data_config,
        args.batch_size,
        device,
        args.num_samples,
        args.sample_seed,
    )

    selected_ids = bundle.sales_frame["id"].astype(str).tolist()
    actuals = load_holdout_actuals(Path(args.data_dir), args.sales_file, selected_ids, args.prediction_length)
    
    all_summary_metrics = {}

    for mode, predictions in forecasts_dict.items():
        logger.info("Computing metrics for mode: %s", mode)
        holdout_metrics_raw, series_metrics_raw = compute_holdout_metrics(
            predictions,
            actuals,
            bundle.sales_values,
            bundle,
            Path(args.data_dir),
            args.prediction_length,
            compute_wrmsse=compute_wrmsse,
        )

        rounded_predictions = np.rint(predictions).clip(min=0.0).astype(np.float32)
        holdout_metrics_rounded, series_metrics_rounded = compute_holdout_metrics(
            rounded_predictions,
            actuals,
            bundle.sales_values,
            bundle,
            Path(args.data_dir),
            args.prediction_length,
            compute_wrmsse=compute_wrmsse,
        )
        
        # Save per-mode artifacts
        forecasts_path_raw = artifact_dir / f"holdout_forecasts_{mode}.csv"
        forecasts_path_rounded = artifact_dir / f"holdout_forecasts_{mode}_rounded.csv"
        write_forecast_csv(forecasts_path_raw, selected_ids, predictions, actuals, series_metrics_raw)
        write_forecast_csv(
            forecasts_path_rounded,
            selected_ids,
            rounded_predictions,
            actuals,
            series_metrics_rounded,
        )
        
        # Add to global summary
        all_summary_metrics[mode] = {
            "raw": holdout_metrics_raw,
            "rounded": holdout_metrics_rounded,
        }
        
        # Log to W&B with mode prefix
        wandb_log(
            wandb_run,
            {
                **{f"holdout/{mode}/raw/{k}": v for k, v in holdout_metrics_raw.items() if v is not None},
                **{f"holdout/{mode}/rounded/{k}": v for k, v in holdout_metrics_rounded.items() if v is not None},
            },
        )
        wandb_save(wandb_run, forecasts_path_raw)
        wandb_save(wandb_run, forecasts_path_rounded)

    # Save aggregate metrics JSON
    metrics_path = artifact_dir / "holdout_metrics_all_modes.json"
    metrics_text = json.dumps(all_summary_metrics, indent=2)
    metrics_path.write_text(metrics_text, encoding="utf-8")
    # Keep the legacy filename in sync for older experiment scripts.
    legacy_metrics_path = artifact_dir / "holdout_metrics.json"
    legacy_metrics_path.write_text(metrics_text, encoding="utf-8")
    wandb_save(wandb_run, metrics_path)
    wandb_save(wandb_run, legacy_metrics_path)

    if compute_wrmsse and "wrmsse" in all_summary_metrics.get("mean", {}).get("raw", {}):
        raw_wrmsse = all_summary_metrics["mean"]["raw"]["wrmsse"]
        rounded_wrmsse = all_summary_metrics["mean"]["rounded"].get("wrmsse")
        if rounded_wrmsse is not None:
            logger.info(
                "Multi-mode holdout complete. Summary (mean WRMSSE raw/rounded): %.5f / %.5f",
                raw_wrmsse,
                rounded_wrmsse,
            )
        else:
            logger.info("Multi-mode holdout complete. Summary (mean WRMSSE): %.5f", raw_wrmsse)
    else:
        logger.info("Multi-mode holdout complete. WRMSSE was not computed.")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for DeepAR training runs."""

    parser = argparse.ArgumentParser(description="Train a from-scratch DeepAR model on M5 data.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--sales-file", default="sales_train_validation.csv")
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
    parser.add_argument("--eval-wrmsse", action="store_true", help="Also compute and save WRMSSE holdout metrics.")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    add_wandb_args(parser)
    return parser


from datetime import datetime

def generate_run_name(args: argparse.Namespace) -> str:
    """Create a unique string identifying this run's configuration and timing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"run_{timestamp}_"
        f"sub{args.subset_size}_"
        f"h{args.hidden_size}_"
        f"lr{args.learning_rate}_"
        f"ep{args.epochs}"
    )

def run_training(args: argparse.Namespace, wandb_run=None) -> tuple[DeepAR, list[dict[str, float | int]]]:
    """Execute the full training and optional evaluation lifecycle for one configuration."""

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
    
    # Use the provided artifact_dir. If it's the generic base, add a subfolder.
    # If experiments.py already gave us a specific subfolder, use it as is.
    base_artifact_dir = Path(args.artifact_dir)
    if base_artifact_dir.name.startswith("run_"):
        artifact_dir = base_artifact_dir
    else:
        run_name = generate_run_name(args)
        artifact_dir = base_artifact_dir / run_name
    
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts for this run will be saved to: %s", artifact_dir)

    bundle = load_m5_bundle(data_config)
    sampler = WindowSampler(bundle, args.context_length, args.prediction_length, seed=args.seed)
    device = choose_device(args.device)

    model = DeepAR(ModelConfig(
        cardinalities=bundle.cardinalities,
        covariate_dim=len(bundle.covariate_columns),
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )).to(device)

    if wandb_run is not None:
        wandb_run.config.update({
            "num_series": bundle.num_series,
            "covariate_dim": len(bundle.covariate_columns),
            "cardinalities": bundle.cardinalities,
            "device_resolved": str(device),
        }, allow_val_change=True)

    save_artifacts(artifact_dir, bundle, data_config, model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float("inf")
    best_state = None
    metrics_history = []

    logger.info("Training on %s series, device=%s", bundle.num_series, device)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, sampler, optimizer, device, epoch,
            args.steps_per_epoch, args.batch_size, args.grad_clip, wandb_run
        )
        val_loss = evaluate(model, sampler, args.batch_size, device)
        
        logger.info("epoch=%s train_nll=%.5f val_nll=%.5f", epoch, train_loss, val_loss)
        wandb_log(wandb_run, {"train/epoch_nll": train_loss, "validation/nll": val_loss, "train/epoch": epoch}, 
                    step=epoch * args.steps_per_epoch)

        metrics_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_checkpoint(artifact_dir / "latest.pt", model, optimizer, data_config, bundle, epoch, best_val_loss, args)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(artifact_dir / "best.pt", model, optimizer, data_config, bundle, epoch, best_val_loss, args)

        (artifact_dir / "metrics.json").write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
        wandb_save(wandb_run, artifact_dir / "metrics.json")
        wandb_save(wandb_run, artifact_dir / "best.pt")

    if args.eval_holdout:
        if best_state is not None:
            model.load_state_dict(best_state)
        run_holdout_evaluation(
            model,
            bundle,
            data_config,
            artifact_dir,
            args,
            device,
            wandb_run,
            compute_wrmsse=args.eval_wrmsse,
        )
    
    return model, metrics_history


def main(argv: list[str] | None = None) -> None:
    """Train DeepAR on sampled M5 windows and write checkpoints/artifacts."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    wandb_run = init_wandb(args, config=vars(args))
    if wandb_run is not None:
        for key, value in wandb_run.config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        logger.info("Updated arguments from W&B config: %s", wandb_run.config)

    try:
        run_training(args, wandb_run)
    finally:
        wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
