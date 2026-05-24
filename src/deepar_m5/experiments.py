import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import pandas as pd

from .train import configure_logging, run_training
from .wandb_utils import add_wandb_args, init_wandb, wandb_finish, wandb_save


logger = logging.getLogger(__name__)


def run_name(index: int, params: dict[str, int | float | str]) -> str:
    """Build a compact directory name for one experiment configuration."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pieces = [
        f"run_{timestamp}_{index:03d}",
        f"subset{params['subset_size']}",
        f"ctx{params['context_length']}",
        f"h{params['hidden_size']}",
        f"emb{params['embedding_dim']}",
        f"ep{params['epochs']}",
        f"steps{params['steps_per_epoch']}",
    ]
    forecast_mode = params.get("forecast_mode")
    if forecast_mode is not None:
        pieces.append(str(forecast_mode).replace("-", ""))
        if forecast_mode == "quantile" and "quantile" in params:
            pieces.append(f"q{int(float(params['quantile']) * 100):02d}")
    return "_".join(pieces)


# Define your hyperparameter grid here for sweeps
GRID_CONFIG = {
    "subset_size": [34590],
    "context_length": [84, 112],
    "batch_size": [1024],
    "epochs": [20],
    "steps_per_epoch": [100],
    "hidden_size": [16],
    "embedding_dim": [8],
    "num_layers": [1,2],
    "dropout": [0.0],
    "learning_rate": [0.001],
    "seed": [42],
}

def experiment_grid() -> list[dict[str, int | float | str]]:
    """Generate experiment configurations from the hardcoded GRID_CONFIG."""

    keys = list(GRID_CONFIG.keys())
    values = list(GRID_CONFIG.values())
    grid = []
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        grid.append(params)
        
    return grid


def flatten_holdout_metrics(all_modes_metrics: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, float | int]:
    """Flatten nested raw/rounded holdout metrics into CSV-friendly columns."""

    flat: dict[str, float | int] = {}
    for mode, variants in all_modes_metrics.items():
        for variant_name, metrics in variants.items():
            for metric_name, value in metrics.items():
                flat[f"{mode}_{variant_name}_{metric_name}"] = value
    return flat


def build_parser() -> argparse.ArgumentParser:
    """Define basic CLI arguments for experiment sweeps."""

    parser = argparse.ArgumentParser(description="Run DeepAR M5 grid search experiments.")
    parser.add_argument("--data-dir", default="m5-forecasting-accuracy")
    parser.add_argument("--sales-file", default="sales_train_validation.csv")
    parser.add_argument("--output-dir", default="artifacts/deepar_m5_experiments")
    parser.add_argument("--prediction-length", type=int, default=28)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--eval-wrmsse", action="store_true", help="Also compute and save WRMSSE holdout metrics.")
    add_wandb_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run an experiment sweep and write per-run plus summary artifacts."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    wandb_group = args.wandb_group
    if "--wandb-group" not in raw_argv:
        wandb_group = f"exp_run_{sweep_timestamp}"
    elif wandb_group in (None, ""):
        wandb_group = f"exp_run_{sweep_timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = experiment_grid()
    summary_rows = []
    logger.info("Running %s experiment configurations from hardcoded grid", len(grid))
    
    for run_idx, params in enumerate(grid, start=1):
        # Create a unique subfolder name for this specific grid combination
        current_run_name = run_name(run_idx, params)
        artifact_dir = output_dir / current_run_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge grid params with system args into a format run_training expects
        train_args = SimpleNamespace(
            **vars(args),
            **params,
            artifact_dir=str(artifact_dir), # Pass this subfolder as the base
            eval_holdout=True, # Experiments always run holdout eval
        )
        
        wandb_run = init_wandb(
            args,
            config=vars(train_args),
            run_name=args.wandb_run_name or artifact_dir.name,
            group=wandb_group,
        )
        
        try:
            # REUSING THE REFACTORED run_training FROM train.py
            model, train_metrics = run_training(train_args, wandb_run)
            
            # Read the metrics file saved by run_training to get final scores
            metrics_path = artifact_dir / "holdout_metrics_all_modes.json"
            if not metrics_path.exists():
                metrics_path = artifact_dir / "holdout_metrics.json"
            holdout_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            flat_holdout_metrics = flatten_holdout_metrics(holdout_metrics)

            row = {
                "run": artifact_dir.name,
                **params,
                **flat_holdout_metrics,
                "best_internal_val_nll": min(metric["val_loss"] for metric in train_metrics),
            }
            summary_rows.append(row)
            summary_path = output_dir / f"summary_{sweep_timestamp}.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            wandb_save(wandb_run, summary_path)
            logger.info("%s holdout metrics: %s", artifact_dir.name, holdout_metrics)
        finally:
            wandb_finish(wandb_run)

    logger.info("Wrote experiment summary to %s", output_dir / f"summary_{sweep_timestamp}.csv")


if __name__ == "__main__":
    main()
