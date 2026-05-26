from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from .data import DataConfig, WindowSampler, load_m5_bundle
from .model import DeepAR, ModelConfig
from .train import batch_to_torch, choose_device, configure_logging


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for a small end-to-end sanity check."""

    parser = argparse.ArgumentParser(description="Smoke-test the from-scratch DeepAR M5 pipeline.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--sales-file", default="sales_train_evaluation.csv")
    parser.add_argument("--subset-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=28)
    parser.add_argument("--prediction-length", type=int, default=7)
    parser.add_argument("--loss", choices=["negative-binomial", "tweedie"], default="negative-binomial")
    parser.add_argument("--tweedie-power", type=float, default=1.5)
    parser.add_argument("--tweedie-dispersion", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Verify data loading, forward pass, loss, backprop, and prediction shapes.

    This runs a tiny end-to-end path: load a selected M5 subset, sample one
    training window batch, compute Negative Binomial distribution parameters,
    backpropagate one optimizer step, and decode a small inference batch.
    """

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = DataConfig(
        data_dir=args.data_dir,
        sales_file=args.sales_file,
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
            distribution=args.loss,
            tweedie_power=args.tweedie_power,
            tweedie_dispersion=args.tweedie_dispersion,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = batch_to_torch(sampler.sample_train_batch(args.batch_size), device)
    mu, aux = model(
        batch["target"],
        batch["covariates"],
        batch["static_cats"],
        batch["scale"],
        prior_history=batch.get("prior_history"),
    )
    loss = model.loss(batch["target"], mu, aux, batch["loss_mask"])
    assert torch.isfinite(loss), "loss is not finite"
    assert mu.shape == batch["target"].shape
    if args.loss == "negative-binomial":
        assert aux is not None
        assert aux.shape == batch["target"].shape
    else:
        assert aux is None
    loss.backward()
    optimizer.step()

    infer_batch = batch_to_torch(sampler.make_inference_batch(np.arange(min(3, bundle.num_series))), device)
    pred = model.predict_mean(
        infer_batch["target"],
        infer_batch["covariates"],
        infer_batch["static_cats"],
        infer_batch["scale"],
        context_length=args.context_length,
        prior_history=infer_batch.get("prior_history"),
    )
    assert pred.shape == (min(3, bundle.num_series), args.prediction_length)
    assert torch.all(pred >= 0)
    samples = model.predict_samples(
        infer_batch["target"],
        infer_batch["covariates"],
        infer_batch["static_cats"],
        infer_batch["scale"],
        context_length=args.context_length,
        num_samples=4,
        prior_history=infer_batch.get("prior_history"),
    )
    assert samples.shape == (4, min(3, bundle.num_series), args.prediction_length)
    assert torch.all(samples >= 0)
    logger.info(
        "smoke ok: "
        "series=%s, covariates=%s, loss=%.5f, pred_shape=%s, sample_shape=%s",
        bundle.num_series,
        len(bundle.covariate_columns),
        loss.item(),
        tuple(pred.shape),
        tuple(samples.shape),
    )


if __name__ == "__main__":
    main()
