from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATIC_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
logger = logging.getLogger(__name__)


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
    """Select a reproducible, roughly category-store-balanced pilot subset.

    The M5 file contains one row per item-store time series. For quick runs we
    sample rows instead of loading every series, but we first group by category
    and store so the subset is not accidentally dominated by one part of the
    hierarchy. Any remaining rows needed to hit ``subset_size`` are sampled from
    the rows not already selected.
    """

    if subset_size is None or subset_size <= 0 or subset_size >= len(frame):
        logger.info("Using all %s series from sales frame", len(frame))
        return frame.reset_index(drop=True)

    logger.info("Selecting %s of %s series with seed=%s", subset_size, len(frame), seed)
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby(["cat_id", "store_id"], sort=True))
    per_group = max(1, subset_size // max(1, len(groups)))
    logger.debug("Found %s cat_id/store_id groups; sampling up to %s per group", len(groups), per_group)
    selected_indices: list[int] = []

    for _, group in groups:
        take = min(len(group), per_group)
        selected_indices.extend(rng.choice(group.index.to_numpy(), size=take, replace=False).tolist())

    if len(selected_indices) < subset_size:
        logger.debug("Selected %s rows from balanced groups; filling remaining rows randomly", len(selected_indices))
        remaining = np.setdiff1d(frame.index.to_numpy(), np.array(selected_indices, dtype=np.int64))
        take = min(subset_size - len(selected_indices), len(remaining))
        selected_indices.extend(rng.choice(remaining, size=take, replace=False).tolist())

    if len(selected_indices) < subset_size:
        logger.warning("Requested subset_size=%s but only selected %s rows", subset_size, len(selected_indices))
    elif len(selected_indices) > subset_size:
        logger.debug("Oversampled %s candidate rows before trimming to subset_size=%s", len(selected_indices), subset_size)

    selected = frame.loc[selected_indices[:subset_size]].sort_values("id").reset_index(drop=True)
    logger.info("Selected series frame shape=%s", selected.shape)
    return selected


def _common_calendar_covariates(calendar: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build calendar covariates that are identical for all item-store series.

    Weekday and month are encoded with sine/cosine pairs so the model sees them
    as cyclic values. Event columns become binary flags indicating whether an
    event is present on that day.
    """

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
    """Align weekly sell prices to daily rows for the selected item-store pairs.

    M5 prices are weekly by ``wm_yr_wk`` while targets are daily. This function
    builds one daily price row per selected series, fills missing weekly prices
    with that series' median available price, and returns both ``log1p(price)``
    and a binary missing-price indicator.
    """

    logger.debug("Raw sell_prices shape=%s", prices.shape)
    calendar_weeks = calendar["wm_yr_wk"].to_numpy()
    log_prices = np.zeros((len(sales_frame), len(calendar)), dtype=np.float32)
    missing = np.ones_like(log_prices, dtype=np.float32)
    selected_keys = sales_frame[["store_id", "item_id"]].drop_duplicates()
    prices = prices.merge(selected_keys, on=["store_id", "item_id"], how="inner")
    logger.debug(
        "Price alignment inputs: calendar_weeks=%s, sales_frame=%s, selected_keys=%s, filtered_prices=%s",
        len(calendar_weeks),
        sales_frame.shape,
        selected_keys.shape,
        prices.shape,
    )

    price_groups = {
        key: group[["wm_yr_wk", "sell_price"]].to_numpy()
        for key, group in prices.groupby(["store_id", "item_id"], sort=False)
    }
    logger.debug(
        "Built %s price lookup groups; sample_keys=%s",
        len(price_groups),
        list(price_groups.keys())[:5],
    )
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

    logger.debug(
        "Price covariate arrays: log_prices=%s, missing=%s, missing_rate=%.4f",
        log_prices.shape,
        missing.shape,
        float(missing.mean()),
    )
    return log_prices, missing


def _build_covariate_cube(
    sales_frame: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Create the full ``series x day x feature`` known-covariate tensor.

    The model receives known future features alongside historical targets. This
    combines calendar features shared by every series, a state-specific SNAP
    indicator, and item-store price features into one cube aligned to the M5 day
    index.
    """

    common, names = _common_calendar_covariates(calendar)
    logger.debug("Common calendar covariates before broadcast=%s", common.shape)
    common = np.broadcast_to(common[None, :, :], (len(sales_frame), common.shape[0], common.shape[1]))
    logger.debug("Common calendar covariates after broadcast=%s", common.shape)
    snap = np.zeros((len(sales_frame), len(calendar), 1), dtype=np.float32)
    logger.debug("SNAP covariate shape=%s", snap.shape)
    for row_idx, state_id in enumerate(sales_frame["state_id"].astype(str)):
        snap_column = f"snap_{state_id}"
        snap[row_idx, :, 0] = calendar[snap_column].astype(np.float32).to_numpy()

    log_prices, price_missing = _build_price_covariates(sales_frame, calendar, prices)
    price_covariates = np.stack([log_prices, price_missing], axis=2)
    covariates = np.concatenate([common.astype(np.float32), snap, price_covariates], axis=2)
    covariate_names = names + ["snap_state", "log_sell_price", "price_missing"]
    logger.info("Built covariate cube shape=%s with features=%s", covariates.shape, covariate_names)
    logger.debug("Price covariates shape=%s; final covariates dtype=%s", price_covariates.shape, covariates.dtype)
    return covariates.astype(np.float32), covariate_names


def _series_scales(values: np.ndarray, train_end: int) -> np.ndarray:
    """Compute per-series demand scales from history before the validation split.

    Each series can have a very different sales level. The scale is based on the
    mean positive demand before validation, with a floor of 1.0, so the model can
    operate on normalized targets and later map predictions back to sales units.
    """

    scales = np.ones(values.shape[0], dtype=np.float32)
    history = values[:, :train_end]
    for idx, series in enumerate(history):
        positive = series[series > 0]
        scale = positive.mean() if len(positive) else series.mean()
        scales[idx] = float(max(scale, 1.0))
    logger.debug(
        "Series scales shape=%s min=%.4f median=%.4f max=%.4f",
        scales.shape,
        float(scales.min()),
        float(np.median(scales)),
        float(scales.max()),
    )
    return scales


def load_m5_bundle(
    config: DataConfig,
    encoders: dict[str, dict[str, int]] | None = None,
    series_ids: list[str] | None = None,
    load_covariates: bool = True,
) -> M5Bundle:
    """Load M5 CSVs and return model-ready arrays plus metadata.

    The loader performs the full data-preparation pipeline: select item-store
    series, sort daily target columns, encode static categorical ids, build
    known calendar/price covariates, compute per-series scales, and return a
    bundle whose arrays are already aligned by series and day.

    When ``series_ids`` is provided, the function preserves that exact order so
    inference predictions line up with the checkpoint's selected series.
    """

    data_dir = Path(config.data_dir)
    sales_path = data_dir / config.sales_file
    if not sales_path.exists():
        sales_path = data_dir / "sales_train_validation.csv"

    logger.info("Loading sales data from %s", sales_path)
    sales = pd.read_csv(sales_path)
    day_columns = find_day_columns(sales.columns)
    logger.info("Raw sales frame shape=%s; day_columns=%s", sales.shape, len(day_columns))
    if series_ids is not None:
        wanted = set(series_ids)
        sales = sales[sales["id"].isin(wanted)].copy()
        sales["_order"] = sales["id"].map({series_id: idx for idx, series_id in enumerate(series_ids)})
        sales = sales.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        logger.info("Loaded %s checkpoint-selected series; sales frame shape=%s", len(series_ids), sales.shape)
    else:
        sales = select_series(sales, config.subset_size, config.seed)

    calendar = pd.read_csv(data_dir / "calendar.csv").sort_values("d").reset_index(drop=True)
    calendar["_day_num"] = calendar["d"].map(day_number)
    calendar = calendar.sort_values("_day_num").drop(columns=["_day_num"]).reset_index(drop=True)
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    logger.info("Calendar shape=%s; sell_prices shape=%s", calendar.shape, prices.shape)

    if encoders is None:
        # Fit on the full metadata so inference has stable ids even if training uses a subset.
        full_sales = pd.read_csv(sales_path, usecols=["id", *STATIC_COLUMNS])
        encoders = fit_encoders(full_sales)
        logger.debug("Fitted static encoders with cardinalities=%s", [len(mapping) for mapping in encoders.values()])

    sales_values = sales[day_columns].to_numpy(dtype=np.float32)
    train_end = max(1, sales_values.shape[1] - config.prediction_length)
    scales = _series_scales(sales_values, train_end=train_end)
    static_cats = encode_static(sales, encoders)
    
    if load_covariates:
        covariates, covariate_columns = _build_covariate_cube(sales, calendar, prices)
    else:
        covariates = np.zeros((0, 0, 0), dtype=np.float32)
        covariate_columns = []

    logger.info(
        "Prepared M5 arrays: sales_values=%s, static_cats=%s, scales=%s, covariates=%s",
        sales_values.shape,
        static_cats.shape,
        scales.shape,
        covariates.shape,
    )
    logger.debug(
        "Validation split: known_days=%s, train_end=%s, context_length=%s, prediction_length=%s",
        sales_values.shape[1],
        train_end,
        config.context_length,
        config.prediction_length,
    )

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
    """Sample train/validation/inference windows without materializing all windows.

    Training examples are random rolling windows from the training portion of
    each selected series. Validation uses one deterministic final holdout window
    per series. Inference uses the latest context window and pads the future
    target positions with zeros before autoregressive decoding.
    """

    def __init__(self, bundle: M5Bundle, context_length: int, prediction_length: int, seed: int = 42):
        """Create a sampler for fixed-length context-plus-prediction sequences."""

        self.bundle = bundle
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sequence_length = context_length + prediction_length
        self.rng = np.random.default_rng(seed)
        self.train_end = bundle.known_days - prediction_length
        self._logged_train_batch = False
        self._logged_validation_batch = False
        self._logged_inference_batch = False
        if self.train_end < self.sequence_length:
            raise ValueError("Not enough known days for the requested context and prediction lengths.")
        logger.info(
            "WindowSampler: num_series=%s known_days=%s train_end=%s context=%s prediction=%s sequence=%s",
            bundle.num_series,
            bundle.known_days,
            self.train_end,
            context_length,
            prediction_length,
            self.sequence_length,
        )

    def sample_train_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample random rolling windows from the training portion of the data.

        Each row in the returned batch independently picks a series and a start
        day before the validation cutoff, then extracts
        ``context_length + prediction_length`` aligned target/covariate values.
        """

        series_idx = self.rng.integers(0, self.bundle.num_series, size=batch_size)
        max_start = self.train_end - self.sequence_length
        starts = self.rng.integers(0, max_start + 1, size=batch_size)
        batch = self._make_batch(series_idx, starts)
        if not self._logged_train_batch:
            logger.debug(
                "Sampled first train batch: series_idx_shape=%s starts_shape=%s start_min=%s start_max=%s",
                series_idx.shape,
                starts.shape,
                int(starts.min()),
                int(starts.max()),
            )
            self._log_batch_shapes("train", batch)
            self._logged_train_batch = True
        return batch

    def iter_validation_batches(self, batch_size: int) -> Iterable[dict[str, np.ndarray]]:
        """Yield deterministic validation windows ending at the last known day.

        The window starts ``context_length`` days before ``train_end``. The loss
        mask later ignores the context portion and scores only the final
        ``prediction_length`` days.
        """

        start = self.train_end - self.context_length
        starts = np.full((self.bundle.num_series,), start, dtype=np.int64)
        all_indices = np.arange(self.bundle.num_series)
        for offset in range(0, self.bundle.num_series, batch_size):
            batch = self._make_batch(all_indices[offset : offset + batch_size], starts[offset : offset + batch_size])
            if not self._logged_validation_batch:
                logger.debug("First validation batch uses start=%s and offset=%s", int(start), int(offset))
                self._log_batch_shapes("validation", batch)
                self._logged_validation_batch = True
            yield batch

    def make_inference_batch(self, series_idx: np.ndarray) -> dict[str, np.ndarray]:
        """Build context windows for autoregressive future decoding.

        The target array contains real observations for the context portion and
        zeros for future positions because those future targets are unknown. The
        model uses known covariates for the full context-plus-horizon window.
        """

        forecast_start = self.bundle.known_days
        start = forecast_start - self.context_length
        positions = start + np.arange(self.sequence_length)
        targets = np.zeros((len(series_idx), self.sequence_length), dtype=np.float32)
        targets[:, : self.context_length] = self.bundle.sales_values[series_idx, start:forecast_start]

        # Extract target from the day before the window starts (prior_target)
        # If start == 0, there is no prior history, so we use zero.
        if start > 0:
            prior_target = self.bundle.sales_values[series_idx, start - 1]
        else:
            prior_target = np.zeros(len(series_idx), dtype=np.float32)

        batch = {
            "target": targets,
            "prior_target": prior_target,
            "covariates": self.bundle.covariates[series_idx[:, None], positions[None, :]],
            "static_cats": self.bundle.static_cats[series_idx],
            "scale": self.bundle.scales[series_idx, None],
            "loss_mask": np.zeros_like(targets, dtype=np.float32),
            "series_idx": series_idx,
        }
        if not self._logged_inference_batch:
            logger.debug(
                "First inference batch: forecast_start=%s context_start=%s positions_shape=%s",
                int(forecast_start),
                int(start),
                positions.shape,
            )
            self._log_batch_shapes("inference", batch)
            self._logged_inference_batch = True
        return batch

    def _make_batch(self, series_idx: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
        """Assemble target, covariate, static, scale, and mask arrays for windows.

        ``positions`` is a two-dimensional index matrix, one row per sampled
        series. Advanced NumPy indexing then gathers the aligned sales targets
        and known covariates. The mask is zero over the context and one over the
        horizon so likelihood loss is applied only to forecast days.
        """

        offsets = np.arange(self.sequence_length)
        positions = starts[:, None] + offsets[None, :]
        targets = self.bundle.sales_values[series_idx[:, None], positions]

        # Extract target from the day before each window starts (prior_target)
        # For each sample in the batch, check if its start index allows for a prior day.
        prior_target = np.zeros(len(series_idx), dtype=np.float32)
        for i, (s_idx, start) in enumerate(zip(series_idx, starts)):
            if start > 0:
                prior_target[i] = self.bundle.sales_values[s_idx, start - 1]

        loss_mask = np.zeros_like(targets, dtype=np.float32)
        loss_mask[:, self.context_length :] = 1.0
        return {
            "target": targets.astype(np.float32),
            "prior_target": prior_target.astype(np.float32),
            "covariates": self.bundle.covariates[series_idx[:, None], positions].astype(np.float32),
            "static_cats": self.bundle.static_cats[series_idx].astype(np.int64),
            "scale": self.bundle.scales[series_idx, None].astype(np.float32),
            "loss_mask": loss_mask,
            "series_idx": series_idx.astype(np.int64),
        }

    def _log_batch_shapes(self, name: str, batch: dict[str, np.ndarray]) -> None:
        """Log the tensor-like shapes in a sampled batch for debugging."""

        logger.debug(
            "%s batch shapes: target=%s prior_target=%s covariates=%s static_cats=%s scale=%s loss_mask=%s series_idx=%s",
            name,
            batch["target"].shape,
            batch.get("prior_target", np.array([])).shape,
            batch["covariates"].shape,
            batch["static_cats"].shape,
            batch["scale"].shape,
            batch["loss_mask"].shape,
            batch["series_idx"].shape,
        )


def config_to_dict(config: DataConfig) -> dict:
    """Convert a data config dataclass to a serializable dictionary."""

    return asdict(config)
