from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATIC_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class DataConfig:
    """Runtime settings that define how M5 data is loaded and windowed."""

    data_dir: str = "m5-forecasting-accuracy"
    subset_size: int | None = 1000
    context_length: int = 56
    prediction_length: int = 28
    seed: int = 42
    sales_file: str = "sales_train_evaluation.csv"


@dataclass
class M5Bundle:
    """In-memory representation shared by training, validation, and inference."""

    sales_frame: pd.DataFrame
    calendar_frame: pd.DataFrame
    sales_values: np.ndarray
    covariates: np.ndarray
    static_cats: np.ndarray
    scales: np.ndarray
    day_columns: list[str]
    encoders: dict[str, dict[str, int]]
    covariate_columns: list[str]

    @property
    def num_series(self) -> int:
        """Return the number of item-store series in the loaded subset."""

        return int(self.sales_values.shape[0])

    @property
    def known_days(self) -> int:
        """Return the number of observed target days in the sales matrix."""

        return int(self.sales_values.shape[1])

    @property
    def cardinalities(self) -> list[int]:
        """Return embedding cardinalities for each static categorical encoder."""

        return [max(mapping.values(), default=0) + 1 for mapping in self.encoders.values()]


def day_number(day_name: str) -> int:
    """Convert an M5 day label such as ``d_1941`` to its integer index."""

    return int(day_name.split("_", 1)[1])


def find_day_columns(columns: Iterable[str]) -> list[str]:
    """Find and numerically sort daily sales columns from an M5 sales file."""

    return sorted(
        [col for col in columns if col.startswith("d_") and col[2:].isdigit()],
        key=day_number,
    )


def fit_encoders(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Fit integer encoders for static categorical metadata columns."""

    encoders: dict[str, dict[str, int]] = {}
    for column in STATIC_COLUMNS:
        values = sorted(frame[column].astype(str).unique().tolist())
        encoders[column] = {"__UNK__": 0, **{value: idx + 1 for idx, value in enumerate(values)}}
    return encoders


def encode_static(frame: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> np.ndarray:
    """Encode static metadata columns into a dense integer matrix."""

    encoded = []
    for column in STATIC_COLUMNS:
        mapping = encoders[column]
        encoded.append(frame[column].astype(str).map(mapping).fillna(0).astype(np.int64).to_numpy())
    return np.stack(encoded, axis=1)


def save_json(path: Path, payload: dict) -> None:
    """Write JSON metadata with parent-directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Load a JSON metadata file."""

    return json.loads(path.read_text(encoding="utf-8"))


def select_series(frame: pd.DataFrame, subset_size: int | None, seed: int) -> pd.DataFrame:
    """Select a reproducible, roughly category-store-balanced pilot subset."""

    if subset_size is None or subset_size <= 0 or subset_size >= len(frame):
        return frame.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    groups = list(frame.groupby(["cat_id", "store_id"], sort=True))
    per_group = max(1, subset_size // max(1, len(groups)))
    selected_indices: list[int] = []

    for _, group in groups:
        take = min(len(group), per_group)
        selected_indices.extend(rng.choice(group.index.to_numpy(), size=take, replace=False).tolist())

    if len(selected_indices) < subset_size:
        remaining = np.setdiff1d(frame.index.to_numpy(), np.array(selected_indices, dtype=np.int64))
        take = min(subset_size - len(selected_indices), len(remaining))
        selected_indices.extend(rng.choice(remaining, size=take, replace=False).tolist())

    return frame.loc[selected_indices[:subset_size]].sort_values("id").reset_index(drop=True)


def _common_calendar_covariates(calendar: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build calendar covariates that are identical for all item-store series."""

    wday = calendar["wday"].astype(float).to_numpy()
    month = calendar["month"].astype(float).to_numpy()
    event_1 = calendar["event_name_1"].notna().astype(float).to_numpy()
    event_2 = calendar["event_name_2"].notna().astype(float).to_numpy()
    event_type_1 = calendar["event_type_1"].notna().astype(float).to_numpy()
    event_type_2 = calendar["event_type_2"].notna().astype(float).to_numpy()

    covariates = np.stack(
        [
            np.sin(2.0 * np.pi * wday / 7.0),
            np.cos(2.0 * np.pi * wday / 7.0),
            np.sin(2.0 * np.pi * month / 12.0),
            np.cos(2.0 * np.pi * month / 12.0),
            event_1,
            event_2,
            event_type_1,
            event_type_2,
        ],
        axis=1,
    ).astype(np.float32)
    names = [
        "wday_sin",
        "wday_cos",
        "month_sin",
        "month_cos",
        "event_name_1_present",
        "event_name_2_present",
        "event_type_1_present",
        "event_type_2_present",
    ]
    return covariates, names


def _build_price_covariates(
    sales_frame: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Align weekly sell prices to daily rows for the selected item-store pairs."""

    calendar_weeks = calendar["wm_yr_wk"].to_numpy()
    log_prices = np.zeros((len(sales_frame), len(calendar)), dtype=np.float32)
    missing = np.ones_like(log_prices, dtype=np.float32)
    selected_keys = sales_frame[["store_id", "item_id"]].drop_duplicates()
    prices = prices.merge(selected_keys, on=["store_id", "item_id"], how="inner")

    price_groups = {
        key: group[["wm_yr_wk", "sell_price"]].to_numpy()
        for key, group in prices.groupby(["store_id", "item_id"], sort=False)
    }

    for row_idx, row in enumerate(sales_frame.itertuples(index=False)):
        key = (row.store_id, row.item_id)
        weekly_rows = price_groups.get(key)
        if weekly_rows is None or len(weekly_rows) == 0:
            continue
        weekly_map = {int(week): float(price) for week, price in weekly_rows}
        daily = np.array([weekly_map.get(int(week), np.nan) for week in calendar_weeks], dtype=np.float32)
        valid = np.isfinite(daily)
        missing[row_idx] = (~valid).astype(np.float32)
        fill_value = float(np.nanmedian(daily)) if valid.any() else 0.0
        daily = np.where(valid, daily, fill_value)
        log_prices[row_idx] = np.log1p(np.maximum(daily, 0.0)).astype(np.float32)

    return log_prices, missing


def _build_covariate_cube(
    sales_frame: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Create the full ``series x day x feature`` known-covariate tensor."""

    common, names = _common_calendar_covariates(calendar)
    common = np.broadcast_to(common[None, :, :], (len(sales_frame), common.shape[0], common.shape[1]))

    snap = np.zeros((len(sales_frame), len(calendar), 1), dtype=np.float32)
    for row_idx, state_id in enumerate(sales_frame["state_id"].astype(str)):
        snap_column = f"snap_{state_id}"
        snap[row_idx, :, 0] = calendar[snap_column].astype(np.float32).to_numpy()

    log_prices, price_missing = _build_price_covariates(sales_frame, calendar, prices)
    price_covariates = np.stack([log_prices, price_missing], axis=2)
    covariates = np.concatenate([common.astype(np.float32), snap, price_covariates], axis=2)
    return covariates.astype(np.float32), names + ["snap_state", "log_sell_price", "price_missing"]


def _series_scales(values: np.ndarray, train_end: int) -> np.ndarray:
    """Compute per-series demand scales from history before the validation split."""

    scales = np.ones(values.shape[0], dtype=np.float32)
    history = values[:, :train_end]
    for idx, series in enumerate(history):
        positive = series[series > 0]
        scale = positive.mean() if len(positive) else series.mean()
        scales[idx] = float(max(scale, 1.0))
    return scales


def load_m5_bundle(
    config: DataConfig,
    encoders: dict[str, dict[str, int]] | None = None,
    series_ids: list[str] | None = None,
) -> M5Bundle:
    """Load M5 CSVs and return model-ready arrays plus metadata.

    When ``series_ids`` is provided, the function preserves that exact order so
    inference predictions line up with the checkpoint's selected series.
    """

    data_dir = Path(config.data_dir)
    sales_path = data_dir / config.sales_file
    if not sales_path.exists():
        sales_path = data_dir / "sales_train_validation.csv"

    sales = pd.read_csv(sales_path)
    day_columns = find_day_columns(sales.columns)
    if series_ids is not None:
        wanted = set(series_ids)
        sales = sales[sales["id"].isin(wanted)].copy()
        sales["_order"] = sales["id"].map({series_id: idx for idx, series_id in enumerate(series_ids)})
        sales = sales.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    else:
        sales = select_series(sales, config.subset_size, config.seed)

    calendar = pd.read_csv(data_dir / "calendar.csv").sort_values("d").reset_index(drop=True)
    calendar["_day_num"] = calendar["d"].map(day_number)
    calendar = calendar.sort_values("_day_num").drop(columns=["_day_num"]).reset_index(drop=True)
    prices = pd.read_csv(data_dir / "sell_prices.csv")

    if encoders is None:
        # Fit on the full metadata so inference has stable ids even if training uses a subset.
        full_sales = pd.read_csv(sales_path, usecols=["id", *STATIC_COLUMNS])
        encoders = fit_encoders(full_sales)

    sales_values = sales[day_columns].to_numpy(dtype=np.float32)
    train_end = max(1, sales_values.shape[1] - config.prediction_length)
    scales = _series_scales(sales_values, train_end=train_end)
    static_cats = encode_static(sales, encoders)
    covariates, covariate_columns = _build_covariate_cube(sales, calendar, prices)

    return M5Bundle(
        sales_frame=sales,
        calendar_frame=calendar,
        sales_values=sales_values,
        covariates=covariates,
        static_cats=static_cats,
        scales=scales,
        day_columns=day_columns,
        encoders=encoders,
        covariate_columns=covariate_columns,
    )


class WindowSampler:
    """Sample train/validation/inference windows without materializing all windows."""

    def __init__(self, bundle: M5Bundle, context_length: int, prediction_length: int, seed: int = 42):
        """Create a sampler for fixed-length context-plus-prediction sequences."""

        self.bundle = bundle
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sequence_length = context_length + prediction_length
        self.rng = np.random.default_rng(seed)
        self.train_end = bundle.known_days - prediction_length
        if self.train_end < self.sequence_length:
            raise ValueError("Not enough known days for the requested context and prediction lengths.")

    def sample_train_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample random rolling windows from the training portion of the data."""

        series_idx = self.rng.integers(0, self.bundle.num_series, size=batch_size)
        max_start = self.train_end - self.sequence_length
        starts = self.rng.integers(0, max_start + 1, size=batch_size)
        return self._make_batch(series_idx, starts)

    def iter_validation_batches(self, batch_size: int) -> Iterable[dict[str, np.ndarray]]:
        """Yield deterministic validation windows ending at the last known day."""

        start = self.train_end - self.context_length
        starts = np.full((self.bundle.num_series,), start, dtype=np.int64)
        all_indices = np.arange(self.bundle.num_series)
        for offset in range(0, self.bundle.num_series, batch_size):
            yield self._make_batch(all_indices[offset : offset + batch_size], starts[offset : offset + batch_size])

    def make_inference_batch(self, series_idx: np.ndarray) -> dict[str, np.ndarray]:
        """Build context windows for autoregressive future decoding."""

        forecast_start = self.bundle.known_days
        start = forecast_start - self.context_length
        positions = start + np.arange(self.sequence_length)
        targets = np.zeros((len(series_idx), self.sequence_length), dtype=np.float32)
        targets[:, : self.context_length] = self.bundle.sales_values[series_idx, start:forecast_start]
        return {
            "target": targets,
            "covariates": self.bundle.covariates[series_idx[:, None], positions[None, :]],
            "static_cats": self.bundle.static_cats[series_idx],
            "scale": self.bundle.scales[series_idx, None],
            "loss_mask": np.zeros_like(targets, dtype=np.float32),
            "series_idx": series_idx,
        }

    def _make_batch(self, series_idx: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
        """Assemble target, covariate, static, scale, and mask arrays for windows."""

        offsets = np.arange(self.sequence_length)
        positions = starts[:, None] + offsets[None, :]
        targets = self.bundle.sales_values[series_idx[:, None], positions]
        loss_mask = np.zeros_like(targets, dtype=np.float32)
        loss_mask[:, self.context_length :] = 1.0
        return {
            "target": targets.astype(np.float32),
            "covariates": self.bundle.covariates[series_idx[:, None], positions].astype(np.float32),
            "static_cats": self.bundle.static_cats[series_idx].astype(np.int64),
            "scale": self.bundle.scales[series_idx, None].astype(np.float32),
            "loss_mask": loss_mask,
            "series_idx": series_idx.astype(np.int64),
        }


def config_to_dict(config: DataConfig) -> dict:
    """Convert a data config dataclass to a serializable dictionary."""

    return asdict(config)
