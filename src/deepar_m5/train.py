from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from .data import DataConfig, WindowSampler, config_to_dict, load_m5_bundle, save_json
from .model import DeepAR, ModelConfig, negative_binomial_nll


def batch_to_torch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a NumPy batch produced by ``WindowSampler`` onto a PyTorch device."""

    return {
        "target": torch.as_tensor(batch["target"], dtype=torch.float32, device=device),
        "covariates": torch.as_tensor(batch["covariates"], dtype=torch.float32, device=device),
        "static_cats": torch.as_tensor(batch["static_cats"], dtype=torch.long, device=device),
        "scale": torch.as_tensor(batch["scale"], dtype=torch.float32, device=device),
        "loss_mask": torch.as_tensor(batch["loss_mask"], dtype=torch.float32, device=device),
    }


def choose_device(device_arg: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise return the requested device."""

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def evaluate(model: DeepAR, sampler: WindowSampler, batch_size: int, device: torch.device) -> float:
    """Evaluate masked validation NLL over deterministic holdout windows."""

    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for batch_np in sampler.iter_validation_batches(batch_size):
            batch = batch_to_torch(batch_np, device)
            mu, alpha = model(batch["target"], batch["covariates"], batch["static_cats"], batch["scale"])
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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Train DeepAR on sampled M5 windows and write checkpoints/artifacts."""

    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_config = DataConfig(
        data_dir=args.data_dir,
        subset_size=args.subset_size,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        seed=args.seed,
    )
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Loading M5 data...")
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

    save_json(artifact_dir / "encoders.json", bundle.encoders)
    save_json(artifact_dir / "data_config.json", config_to_dict(data_config))
    save_json(artifact_dir / "model_config.json", model.to_config_dict())
    bundle.sales_frame[["id", *["item_id", "dept_id", "cat_id", "store_id", "state_id"]]].to_csv(
        artifact_dir / "selected_series.csv",
        index=False,
    )

    print(f"Training on {bundle.num_series} series, device={device}, covariates={len(bundle.covariate_columns)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = trange(args.steps_per_epoch, desc=f"epoch {epoch}", leave=False)
        for _ in progress:
            batch_np = sampler.sample_train_batch(args.batch_size)
            batch = batch_to_torch(batch_np, device)
            optimizer.zero_grad(set_to_none=True)
            mu, alpha = model(batch["target"], batch["covariates"], batch["static_cats"], batch["scale"])
            loss = negative_binomial_nll(batch["target"], mu, alpha, batch["loss_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(args.steps_per_epoch, 1)
        val_loss = evaluate(model, sampler, args.batch_size, device)
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_nll={train_loss:.5f} val_nll={val_loss:.5f}")

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

        (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
