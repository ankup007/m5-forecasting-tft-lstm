from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import DataConfig, WindowSampler, load_m5_bundle
from .model import DeepAR, ModelConfig, negative_binomial_nll
from .train import batch_to_torch, choose_device


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for a small end-to-end sanity check."""

    parser = argparse.ArgumentParser(description="Smoke-test the from-scratch DeepAR M5 pipeline.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--subset-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=28)
    parser.add_argument("--prediction-length", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Verify data loading, forward pass, loss, backprop, and prediction shapes."""

    args = build_parser().parse_args(argv)
    config = DataConfig(
        data_dir=args.data_dir,
        subset_size=args.subset_size,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )
    bundle = load_m5_bundle(config)
    sampler = WindowSampler(bundle, args.context_length, args.prediction_length)
    device = choose_device(args.device)

    model = DeepAR(
        ModelConfig(
            cardinalities=bundle.cardinalities,
            covariate_dim=len(bundle.covariate_columns),
            hidden_size=16,
            embedding_dim=4,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = batch_to_torch(sampler.sample_train_batch(args.batch_size), device)
    mu, alpha = model(batch["target"], batch["covariates"], batch["static_cats"], batch["scale"])
    loss = negative_binomial_nll(batch["target"], mu, alpha, batch["loss_mask"])
    assert torch.isfinite(loss), "loss is not finite"
    assert mu.shape == batch["target"].shape
    assert alpha.shape == batch["target"].shape
    loss.backward()
    optimizer.step()

    infer_batch = batch_to_torch(sampler.make_inference_batch(np.arange(min(3, bundle.num_series))), device)
    pred = model.predict_mean(
        infer_batch["target"],
        infer_batch["covariates"],
        infer_batch["static_cats"],
        infer_batch["scale"],
        context_length=args.context_length,
    )
    assert pred.shape == (min(3, bundle.num_series), args.prediction_length)
    assert torch.all(pred >= 0)
    print(
        "smoke ok: "
        f"series={bundle.num_series}, covariates={len(bundle.covariate_columns)}, "
        f"loss={loss.item():.5f}, pred_shape={tuple(pred.shape)}"
    )


if __name__ == "__main__":
    main()
