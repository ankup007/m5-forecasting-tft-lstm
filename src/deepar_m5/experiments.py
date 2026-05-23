import argparse
import itertools
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .train import configure_logging, run_training
from .wandb_utils import add_wandb_args, init_wandb, wandb_finish, wandb_save


logger = logging.getLogger(__name__)


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


# Define your hyperparameter grid here for sweeps
GRID_CONFIG = {
    "subset_size": [100],
    "context_length": [56],
    "batch_size": [32],
    "epochs": [5],
    "steps_per_epoch": [10],
    "hidden_size": [16, 32],
    "embedding_dim": [8],
    "num_layers": [1],
    "dropout": [0.0],
    "learning_rate": [0.01],
    "seed": [42],
    "forecast_mode": ["mean"],
    "quantile": [0.5],
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
    add_wandb_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run an experiment sweep and write per-run plus summary artifacts."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    
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
            group=args.wandb_group or output_dir.name,
        )
        
        try:
            # REUSING THE REFACTORED run_training FROM train.py
            model, train_metrics = run_training(train_args, wandb_run)
            
            # Read the metrics file saved by run_training to get final scores
            metrics_path = artifact_dir / "holdout_metrics.json"
            holdout_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

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
