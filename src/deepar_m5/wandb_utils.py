from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add optional Weights & Biases tracking flags to a CLI parser."""

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="m5-competition")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags.")


def init_wandb(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_name: str | None = None,
    group: str | None = None,
) -> Any:
    """Start a W&B run if enabled; otherwise return ``None``."""

    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Install it with `pip install wandb`.") from exc

    tags = [tag.strip() for tag in getattr(args, "wandb_tags", "").split(",") if tag.strip()]
    return wandb.init(
        entity=getattr(args, "wandb_entity", None),
        project=getattr(args, "wandb_project", "m5-competition"),
        name=run_name or getattr(args, "wandb_run_name", None),
        group=group or getattr(args, "wandb_group", None),
        mode=getattr(args, "wandb_mode", "online"),
        tags=tags,
        config=config,
    )


def wandb_log(run: Any, payload: dict[str, Any], step: int | None = None) -> None:
    """Log metrics when a W&B run is active."""

    if run is None:
        return
    run.log(payload, step=step)


def wandb_save(run: Any, path: str | Path) -> None:
    """Ask W&B to save a local artifact file when a run is active."""

    if run is None:
        return
    try:
        run.save(str(path))
    except Exception as exc:  # pragma: no cover - W&B file syncing is best-effort.
        logger.warning("Could not save %s to W&B: %s", path, exc)


def wandb_finish(run: Any) -> None:
    """Finish a W&B run when one is active."""

    if run is not None:
        run.finish()
