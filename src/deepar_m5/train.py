from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import trange

from .data import DataConfig, WindowSampler, config_to_dict, load_m5_bundle, save_json
from .evaluation import (
    WRMSSEContext,
    compute_holdout_metrics,
    forecast_multi_summaries,
    load_holdout_actuals,
    precompute_wrmsse_contexts,
    write_forecast_csv,
)
from .model import DeepAR, ModelConfig
from .utils import batch_to_torch, choose_device, configure_logging
from .wandb_utils import add_wandb_args, init_wandb, wandb_finish, wandb_log, wandb_save


logger = logging.getLogger(__name__)


def evaluate(
    model: DeepAR,
    sampler: WindowSampler,
    batch_size: int,
    device: torch.device,
) -> float:
    """Evaluate the configured masked loss over deterministic holdout windows."""

    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for batch_np in sampler.iter_validation_batches(batch_size):
            batch = batch_to_torch(batch_np, device)
            mu, aux = model(
                batch["target"],
                batch["covariates"],
                batch["static_cats"],
                batch["scale"],
                prior_history=batch.get("prior_history"),
                initial_zero_counter=batch.get("initial_zero_counter"),
            )
            loss_sum = model.loss(batch["target"], mu, aux, batch["loss_mask"])
            weight = float(batch["loss_mask"].sum().item())
            total_loss += float(loss_sum.item()) * weight
            total_weight += weight
    return total_loss / max(total_weight, 1.0)


def rolled_feedback_probability(args: argparse.Namespace, epoch: int) -> float:
    """Return the scheduled probability of feeding sampled values during training."""

    max_prob = max(0.0, min(1.0, float(getattr(args, "rolled_feedback_max_prob", 0.0))))
    if max_prob <= 0.0:
        return 0.0
    warmup_epochs = max(0, int(getattr(args, "rolled_feedback_warmup_epochs", 0)))
    ramp_epochs = max(1, int(getattr(args, "rolled_feedback_ramp_epochs", 1)))
    if epoch <= warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return max_prob * progress


def rolling_validation_origins(
    sampler: WindowSampler,
    num_origins: int,
    stride: int,
) -> list[int]:
    """Return historical forecast origins with actual future data available."""

    if num_origins <= 0:
        return []
    stride = max(1, int(stride))
    latest_origin = sampler.train_end
    origins = []
    for idx in range(num_origins):
        origin = latest_origin - idx * stride
        if origin < sampler.context_length:
            break
        if origin + sampler.prediction_length > sampler.bundle.known_days:
            continue
        origins.append(origin)
    return origins


@torch.no_grad()
def forecast_origin(
    model: DeepAR,
    sampler: WindowSampler,
    forecast_start: int,
    batch_size: int,
    device: torch.device,
    forecast_mode: str,
    num_samples: int,
    sample_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast one historical origin autoregressively and return predictions plus actuals."""

    if forecast_mode not in {"mean", "sample-mean"}:
        raise ValueError(f"Unknown autoregressive validation forecast mode: {forecast_mode}")
    if sample_seed is not None:
        torch.manual_seed(int(sample_seed) + int(forecast_start))

    model.eval()
    all_indices = np.arange(sampler.bundle.num_series)
    predictions = np.zeros((sampler.bundle.num_series, sampler.prediction_length), dtype=np.float32)
    actuals = sampler.bundle.sales_values[:, forecast_start : forecast_start + sampler.prediction_length].astype(np.float32)

    for offset in range(0, sampler.bundle.num_series, batch_size):
        series_idx = all_indices[offset : offset + batch_size]
        batch_np = sampler.make_prediction_batch(series_idx, forecast_start=forecast_start)
        batch = batch_to_torch(batch_np, device)
        if forecast_mode == "mean":
            pred = model.predict_mean(
                batch["target"],
                batch["covariates"],
                batch["static_cats"],
                batch["scale"],
                context_length=sampler.context_length,
                prior_history=batch.get("prior_history"),
                initial_zero_counter=batch.get("initial_zero_counter"),
            )
        else:
            samples = model.predict_samples(
                batch["target"],
                batch["covariates"],
                batch["static_cats"],
                batch["scale"],
                context_length=sampler.context_length,
                num_samples=num_samples,
                prior_history=batch.get("prior_history"),
                initial_zero_counter=batch.get("initial_zero_counter"),
            )
            expected_shape = (num_samples, len(series_idx), sampler.prediction_length)
            if samples.shape != expected_shape:
                raise ValueError(f"Unexpected sample forecast shape: {tuple(samples.shape)}")
            pred = samples.mean(dim=0)
        predictions[series_idx] = pred.clamp_min(0.0).cpu().numpy()

    return predictions, actuals


def evaluate_autoregressive(
    model: DeepAR,
    sampler: WindowSampler,
    batch_size: int,
    device: torch.device,
    data_dir: Path,
    origins: list[int],
    forecast_mode: str,
    num_samples: int,
    sample_seed: int | None,
    wrmsse_contexts: dict[int, WRMSSEContext],
    wrmsse_level_indices: list[tuple[np.ndarray | None, int | None]],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Evaluate autoregressive forecasts over one or more rolling validation origins."""

    origin_metrics: list[dict[str, float | int]] = []
    for origin in origins:
        predictions, actuals = forecast_origin(
            model,
            sampler,
            forecast_start=origin,
            batch_size=batch_size,
            device=device,
            forecast_mode=forecast_mode,
            num_samples=num_samples,
            sample_seed=sample_seed,
        )
        metrics, _ = compute_holdout_metrics(
            predictions,
            actuals,
            sampler.bundle.sales_values[:, :origin],
            sampler.bundle,
            data_dir,
            sampler.prediction_length,
            compute_wrmsse=True,
            wrmsse_context=wrmsse_contexts[origin],
            wrmsse_level_indices=wrmsse_level_indices,
        )
        metrics = {**metrics, "forecast_start": int(origin)}
        origin_metrics.append(metrics)

    if not origin_metrics:
        return {}, []

    numeric_keys = [
        key
        for key, value in origin_metrics[0].items()
        if key != "forecast_start" and isinstance(value, (int, float)) and np.isfinite(float(value))
    ]
    summary: dict[str, float | int] = {"num_origins": len(origin_metrics)}
    for key in numeric_keys:
        values = [float(metrics[key]) for metrics in origin_metrics if np.isfinite(float(metrics.get(key, np.nan)))]
        if values:
            summary[key] = float(np.mean(values))
    return summary, origin_metrics


def save_checkpoint(
    path: Path,
    model: DeepAR,
    optimizer: torch.optim.Optimizer,
    scheduler,
    data_config: DataConfig,
    bundle,
    epoch: int,
    best_score: float,
    train_args: argparse.Namespace,
    model_state: dict[str, torch.Tensor] | None = None,
    include_optimizer_state: bool = True,
    include_scheduler_state: bool = True,
) -> None:
    """Persist model weights plus all metadata needed for reproducible inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model_state if model_state is not None else model.state_dict(),
            "optimizer_state": optimizer.state_dict() if include_optimizer_state else None,
            "scheduler_state": None if scheduler is None or not include_scheduler_state else scheduler.state_dict(),
            "model_config": model.to_config_dict(),
            "data_config": config_to_dict(data_config),
            "encoders": bundle.encoders,
            "event_encoders": bundle.event_encoders,
            "covariate_columns": bundle.covariate_columns,
            "selected_series_ids": bundle.sales_frame["id"].astype(str).tolist(),
            "epoch": epoch,
            "best_val_loss": best_score,
            "best_score": best_score,
            "checkpoint_metric": getattr(train_args, "checkpoint_metric", "autoreg_wrmsse"),
            "checkpoint_top_k": getattr(train_args, "checkpoint_top_k", 3),
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
    rolled_feedback_prob: float,
    wandb_run=None,
) -> float:
    """Run one training epoch and return the mean masked forecast loss."""

    model.train()
    running_loss = 0.0
    progress = trange(steps_per_epoch, desc=f"epoch {epoch}", leave=True)
    for step_idx in progress:
        global_step = (epoch - 1) * steps_per_epoch + step_idx + 1
        batch_np = sampler.sample_train_batch(batch_size)
        batch = batch_to_torch(batch_np, device)
        optimizer.zero_grad(set_to_none=True)
        mu, aux = model(
            batch["target"],
            batch["covariates"],
            batch["static_cats"],
            batch["scale"],
            prior_history=batch.get("prior_history"),
            initial_zero_counter=batch.get("initial_zero_counter"),
            context_length=sampler.context_length,
            rolled_feedback_prob=rolled_feedback_prob,
        )
        loss = model.loss(batch["target"], mu, aux, batch["loss_mask"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_loss = float(loss.item())
        running_loss += batch_loss
        progress.set_postfix(loss=f"{batch_loss:.4f}", roll=f"{rolled_feedback_prob:.2f}")
        wandb_log(
            wandb_run,
            {
                "train/batch_loss": batch_loss,
                "train/grad_norm": float(grad_norm),
                "train/epoch": epoch,
                "train/rolled_feedback_prob": rolled_feedback_prob,
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
        "prior_history_length": getattr(args, "prior_history_length", data_config.prior_history_length),
        "batch_size": getattr(args, "batch_size", 128),
        "epochs": getattr(args, "epochs", 10),
        "steps_per_epoch": getattr(args, "steps_per_epoch", 200),
        "hidden_size": getattr(args, "hidden_size", 64),
        "num_layers": getattr(args, "num_layers", 1),
        "dropout": getattr(args, "dropout", 0.0),
        "learning_rate": getattr(args, "learning_rate", 1e-3),
        "loss": getattr(args, "loss", "negative-binomial"),
        "tweedie_power": getattr(args, "tweedie_power", 1.5),
        "tweedie_dispersion": getattr(args, "tweedie_dispersion", 1.0),
        "zero_counter_log_divisor": getattr(args, "zero_counter_log_divisor", 4.0),
        "log_scale_divisor": getattr(args, "log_scale_divisor", 5.0),
        "nb_alpha_max": getattr(args, "nb_alpha_max", 1e4),
        "rolled_feedback_max_prob": getattr(args, "rolled_feedback_max_prob", 0.0),
        "rolled_feedback_warmup_epochs": getattr(args, "rolled_feedback_warmup_epochs", 0),
        "rolled_feedback_ramp_epochs": getattr(args, "rolled_feedback_ramp_epochs", 1),
        "autoreg_val_origins": getattr(args, "autoreg_val_origins", 1),
        "autoreg_val_stride": getattr(args, "autoreg_val_stride", data_config.prediction_length),
        "autoreg_val_every": getattr(args, "autoreg_val_every", 1),
        "autoreg_val_mode": getattr(args, "autoreg_val_mode", "mean"),
        "autoreg_val_num_samples": getattr(args, "autoreg_val_num_samples", 50),
        "checkpoint_metric": getattr(args, "checkpoint_metric", "autoreg_wrmsse"),
        "checkpoint_top_k": 3,
        "scheduler": getattr(args, "scheduler", "cosine"),
        "eta_min": getattr(args, "eta_min", 0.0),
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
    wrmsse_context: WRMSSEContext | None = None,
    wrmsse_level_indices: list[tuple[np.ndarray | None, int | None]] | None = None,
    tag_prefix: str = "holdout",
    forecasts_dict: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Run the 12-level competitive evaluation for multiple forecast summaries.
    
    If forecasts_dict is provided, generation is skipped and the provided 
    forecasts are used for metric calculation and artifact saving.
    """

    logger.info("Running multi-mode holdout evaluation into: %s", artifact_dir)
    model.eval()

    # Generate or use provided forecasts
    if forecasts_dict is None:
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
    metric_train_values = bundle.sales_values
    
    all_summary_metrics = {}

    for mode, predictions in forecasts_dict.items():
        logger.info("Computing metrics for mode: %s", mode)
        holdout_metrics_raw, series_metrics_raw = compute_holdout_metrics(
            predictions,
            actuals,
            metric_train_values,
            bundle,
            Path(args.data_dir),
            args.prediction_length,
            compute_wrmsse=compute_wrmsse,
            wrmsse_context=wrmsse_context,
            wrmsse_level_indices=wrmsse_level_indices,
        )

        rounded_predictions = np.rint(predictions).clip(min=0.0).astype(np.float32)
        holdout_metrics_rounded, series_metrics_rounded = compute_holdout_metrics(
            rounded_predictions,
            actuals,
            metric_train_values,
            bundle,
            Path(args.data_dir),
            args.prediction_length,
            compute_wrmsse=compute_wrmsse,
            wrmsse_context=wrmsse_context,
            wrmsse_level_indices=wrmsse_level_indices,
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
                **{f"{tag_prefix}/{mode}/raw/{k}": v for k, v in holdout_metrics_raw.items() if v is not None},
                **{f"{tag_prefix}/{mode}/rounded/{k}": v for k, v in holdout_metrics_rounded.items() if v is not None},
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
    
    return forecasts_dict


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for DeepAR training runs."""

    parser = argparse.ArgumentParser(description="Train a from-scratch DeepAR model on M5 data.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--sales-file", default="sales_train_validation.csv")
    parser.add_argument("--artifact-dir", default="artifacts/deepar_m5")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=56)
    parser.add_argument("--prediction-length", type=int, default=28)
    parser.add_argument("--prior-history-length", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["negative-binomial", "tweedie"], default="negative-binomial")
    parser.add_argument("--tweedie-power", type=float, default=1.5)
    parser.add_argument("--tweedie-dispersion", type=float, default=1.0)
    parser.add_argument("--zero-counter-log-divisor", type=float, default=4.0)
    parser.add_argument("--log-scale-divisor", type=float, default=5.0)
    parser.add_argument("--nb-alpha-max", type=float, default=1e4)
    parser.add_argument("--rolled-feedback-max-prob", type=float, default=0.0)
    parser.add_argument("--rolled-feedback-warmup-epochs", type=int, default=0)
    parser.add_argument("--rolled-feedback-ramp-epochs", type=int, default=1)
    parser.add_argument("--autoreg-val-origins", type=int, default=1)
    parser.add_argument("--autoreg-val-stride", type=int, default=28)
    parser.add_argument("--autoreg-val-every", type=int, default=1)
    parser.add_argument("--autoreg-val-mode", choices=["mean", "sample-mean"], default="mean")
    parser.add_argument("--autoreg-val-num-samples", type=int, default=50)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["teacher_loss", "autoreg_wrmsse", "autoreg_rmsse", "autoreg_wape", "autoreg_spike_day_mae"],
        default="autoreg_wrmsse",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--eta-min", type=float, default=0.0)
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


def generate_run_name(args: argparse.Namespace) -> str:
    """Create a unique string identifying this run's configuration and timing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"run_{timestamp}_"
        f"sub{args.subset_size}_"
        f"h{args.hidden_size}_"
        f"lr{args.learning_rate}_"
        f"loss{args.loss.replace('-', '')}_"
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
        prior_history_length=getattr(args, "prior_history_length", 28),
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

    # In the updated feature set:
    # event_cardinalities come from bundle.event_cardinalities
    # continuous_covariate_dim = total_covariates - num_events
    num_events = len(bundle.event_cardinalities)
    continuous_covariate_dim = len(bundle.covariate_columns) - num_events

    model = DeepAR(ModelConfig(
        cardinalities=bundle.cardinalities,
        event_cardinalities=bundle.event_cardinalities,
        covariate_dim=continuous_covariate_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        distribution=args.loss,
        tweedie_power=args.tweedie_power,
        tweedie_dispersion=args.tweedie_dispersion,
        prior_history_length=data_config.prior_history_length,
        zero_counter_log_divisor=getattr(args, "zero_counter_log_divisor", 4.0),
        log_scale_divisor=getattr(args, "log_scale_divisor", 5.0),
        nb_alpha_max=getattr(args, "nb_alpha_max", 1e4),
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
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.eta_min,
        )
    
    best_score = float("inf")
    best_checkpoint_path = artifact_dir / "best.pt"
    top_k = 3
    top_checkpoint_paths = [artifact_dir / f"best_{rank}.pt" for rank in range(1, top_k + 1)]
    best_checkpoints: list[dict[str, object]] = []
    metrics_history = []
    selection_metric = getattr(args, "checkpoint_metric", "autoreg_wrmsse")
    autoreg_origins = rolling_validation_origins(
        sampler,
        num_origins=int(getattr(args, "autoreg_val_origins", 1)),
        stride=int(getattr(args, "autoreg_val_stride", args.prediction_length)),
    )
    logger.info("Autoregressive validation origins: %s", autoreg_origins)
    wrmsse_level_indices, wrmsse_contexts = precompute_wrmsse_contexts(
        bundle,
        Path(args.data_dir),
        args.prediction_length,
        [*autoreg_origins, bundle.known_days],
    )
    logger.info("Precomputed WRMSSE contexts for forecast origins: %s", sorted(wrmsse_contexts))

    def _clone_model_state() -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    def _save_best_checkpoints() -> None:
        best_checkpoints.sort(key=lambda entry: float(entry["score"]))
        del best_checkpoints[top_k:]
        for rank, path in enumerate(top_checkpoint_paths, start=1):
            if rank > len(best_checkpoints):
                if path.exists():
                    path.unlink()
                continue
            entry = best_checkpoints[rank - 1]
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                data_config,
                bundle,
                int(entry["epoch"]),
                float(entry["score"]),
                args,
                model_state=entry["model_state"],
                include_optimizer_state=False,
                include_scheduler_state=False,
            )
        if best_checkpoints:
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                data_config,
                bundle,
                int(best_checkpoints[0]["epoch"]),
                float(best_checkpoints[0]["score"]),
                args,
                model_state=best_checkpoints[0]["model_state"],
                include_optimizer_state=False,
                include_scheduler_state=False,
            )
        best_manifest = [
            {
                "rank": idx + 1,
                "epoch": int(entry["epoch"]),
                "score": float(entry["score"]),
                "path": path.name,
                "metric": entry["metric"],
            }
            for idx, (entry, path) in enumerate(zip(best_checkpoints, top_checkpoint_paths))
        ]
        (artifact_dir / "best_checkpoints.json").write_text(
            json.dumps(best_manifest, indent=2),
            encoding="utf-8",
        )

    logger.info("Training on %s series, device=%s", bundle.num_series, device)
    for epoch in range(1, args.epochs + 1):
        roll_prob = rolled_feedback_probability(args, epoch)
        train_loss = train_epoch(
            model, sampler, optimizer, device, epoch,
            args.steps_per_epoch, args.batch_size, args.grad_clip,
            roll_prob, wandb_run
        )
        val_loss = evaluate(model, sampler, args.batch_size, device)
        autoreg_summary: dict[str, float | int] = {}
        autoreg_by_origin: list[dict[str, float | int]] = []
        autoreg_every = max(1, int(getattr(args, "autoreg_val_every", 1)))
        if autoreg_origins and (epoch % autoreg_every == 0 or epoch == args.epochs):
            autoreg_summary, autoreg_by_origin = evaluate_autoregressive(
                model,
                sampler,
                args.batch_size,
                device,
                Path(args.data_dir),
                autoreg_origins,
                forecast_mode=getattr(args, "autoreg_val_mode", "mean"),
                num_samples=int(getattr(args, "autoreg_val_num_samples", 50)),
                sample_seed=getattr(args, "sample_seed", None),
                wrmsse_contexts=wrmsse_contexts,
                wrmsse_level_indices=wrmsse_level_indices,
            )
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        if selection_metric == "teacher_loss":
            selection_score = val_loss
            effective_selection_metric = "teacher_loss"
        else:
            selection_scores = {
                "autoreg_wrmsse": float(autoreg_summary.get("wrmsse", float("inf"))),
                "autoreg_rmsse": float(autoreg_summary.get("rmsse", float("inf"))),
                "autoreg_wape": float(autoreg_summary.get("wape", float("inf"))),
                "autoreg_spike_day_mae": float(autoreg_summary.get("spike_day_mae", float("inf"))),
            }
            selection_score = selection_scores.get(selection_metric, float("inf"))
            effective_selection_metric = selection_metric
            if autoreg_summary and not np.isfinite(selection_score):
                selection_score = val_loss
                effective_selection_metric = "teacher_loss"
        finite_selection_score = np.isfinite(selection_score)
        logger.info(
            "epoch=%s train_loss=%.5f val_loss=%.5f autoreg_rmsse=%s score=%s metric=%s roll=%.3f lr=%.8f loss=%s",
            epoch,
            train_loss,
            val_loss,
            f"{autoreg_summary.get('rmsse'):.5f}" if "rmsse" in autoreg_summary else "NA",
            f"{selection_score:.5f}" if finite_selection_score else "NA",
            effective_selection_metric,
            roll_prob,
            current_lr,
            args.loss,
        )
        wandb_payload = {
            "train/epoch_loss": train_loss,
            "validation/loss": val_loss,
            "train/epoch": epoch,
            "train/lr": current_lr,
            "train/rolled_feedback_prob": roll_prob,
            "validation/autoreg_num_origins": int(autoreg_summary.get("num_origins", 0)),
        }
        if finite_selection_score:
            wandb_payload["validation/selection_score"] = selection_score
        wandb_payload.update({f"validation/autoreg_{k}": v for k, v in autoreg_summary.items()})
        if args.loss == "negative-binomial":
            wandb_payload.update({"train/epoch_nll": train_loss, "validation/nll": val_loss})
        wandb_log(wandb_run, wandb_payload, step=epoch * args.steps_per_epoch)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "rolled_feedback_prob": roll_prob,
            "selection_score": selection_score if finite_selection_score else None,
            "selection_metric": effective_selection_metric,
            **{f"autoreg_{k}": v for k, v in autoreg_summary.items()},
        }
        metrics_history.append(epoch_metrics)
        if autoreg_by_origin:
            (artifact_dir / "rolling_validation_metrics.json").write_text(
                json.dumps(autoreg_by_origin, indent=2),
                encoding="utf-8",
            )
        current_cutoff = float("inf") if len(best_checkpoints) < top_k else float(best_checkpoints[-1]["score"])
        if finite_selection_score and (len(best_checkpoints) < top_k or selection_score < current_cutoff):
            best_checkpoints.append(
                {
                    "score": float(selection_score),
                    "epoch": epoch,
                    "metric": effective_selection_metric,
                    "model_state": _clone_model_state(),
                }
            )
            previous_best = best_score
            best_checkpoints.sort(key=lambda entry: float(entry["score"]))
            del best_checkpoints[top_k:]
            best_score = float(best_checkpoints[0]["score"]) if best_checkpoints else float("inf")
            _save_best_checkpoints()
            if autoreg_by_origin and best_score < previous_best:
                (artifact_dir / "best_rolling_validation_metrics.json").write_text(
                    json.dumps(autoreg_by_origin, indent=2),
                    encoding="utf-8",
                )
        save_checkpoint(artifact_dir / "latest.pt", model, optimizer, scheduler, data_config, bundle, epoch, best_score, args)

        (artifact_dir / "metrics.json").write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
        wandb_save(wandb_run, artifact_dir / "metrics.json")
        rolling_metrics_path = artifact_dir / "rolling_validation_metrics.json"
        if rolling_metrics_path.exists():
            wandb_save(wandb_run, rolling_metrics_path)
        best_rolling_metrics_path = artifact_dir / "best_rolling_validation_metrics.json"
        if best_rolling_metrics_path.exists():
            wandb_save(wandb_run, best_rolling_metrics_path)
        best_checkpoints_path = artifact_dir / "best_checkpoints.json"
        if best_checkpoints_path.exists():
            wandb_save(wandb_run, best_checkpoints_path)
        for path in [artifact_dir / "best.pt", *top_checkpoint_paths[1:]]:
            if path.exists():
                wandb_save(wandb_run, path)

    if args.eval_holdout:
        # Load manifest to find all top-k checkpoints
        manifest_path = artifact_dir / "best_checkpoints.json"
        checkpoints_to_eval = []
        if manifest_path.exists():
            try:
                best_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in best_manifest:
                    checkpoints_to_eval.append((entry["rank"], entry["path"]))
            except Exception as e:
                logger.warning("Failed to parse best_checkpoints.json: %s", e)
        
        # Fallback if manifest is missing or empty
        if not checkpoints_to_eval and best_checkpoint_path.exists():
            checkpoints_to_eval.append((1, "best.pt"))

        all_rank_forecasts = []
        for rank, checkpoint_name in checkpoints_to_eval:
            checkpoint_path = artifact_dir / checkpoint_name
            if not checkpoint_path.exists():
                continue
            
            logger.info("Running holdout evaluation for rank %s (%s)", rank, checkpoint_name)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            
            # Use a subfolder for this rank's evaluation
            eval_subdir = artifact_dir / f"eval_rank_{rank}"
            eval_subdir.mkdir(parents=True, exist_ok=True)
            
            rank_forecasts = run_holdout_evaluation(
                model,
                bundle,
                data_config,
                eval_subdir,
                args,
                device,
                wandb_run,
                compute_wrmsse=args.eval_wrmsse,
                wrmsse_context=wrmsse_contexts[bundle.known_days],
                wrmsse_level_indices=wrmsse_level_indices,
                tag_prefix=f"holdout/rank_{rank}",
            )
            all_rank_forecasts.append(rank_forecasts)
            
            # For rank 1, also copy files to the root artifact_dir for backward compatibility and UI
            if rank == 1:
                for f in eval_subdir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, artifact_dir / f.name)

        # 4. Ensemble evaluation (average of top-k)
        if len(all_rank_forecasts) > 1:
            logger.info("Computing ensemble forecast from %s checkpoints", len(all_rank_forecasts))
            ensemble_forecasts = {}
            modes = all_rank_forecasts[0].keys()
            for mode in modes:
                # Stack all forecasts for this mode and take the mean
                stacked = np.stack([f[mode] for f in all_rank_forecasts], axis=0)
                ensemble_forecasts[mode] = np.mean(stacked, axis=0)
            
            ensemble_subdir = artifact_dir / "eval_ensemble"
            ensemble_subdir.mkdir(parents=True, exist_ok=True)
            run_holdout_evaluation(
                model,
                bundle,
                data_config,
                ensemble_subdir,
                args,
                device,
                wandb_run,
                compute_wrmsse=args.eval_wrmsse,
                wrmsse_context=wrmsse_contexts[bundle.known_days],
                wrmsse_level_indices=wrmsse_level_indices,
                tag_prefix="holdout/ensemble",
                forecasts_dict=ensemble_forecasts,
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
