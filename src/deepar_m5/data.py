from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATIC_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_COLUMNS = ["event_name_1", "event_type_1"]
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
    use_event_embeddings: bool = True


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
    event_encoders: dict[str, dict[str, int]] | None = None
    zero_counts: np.ndarray | None = None

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

    @property
    def event_cardinalities(self) -> list[int]:
        """Return cardinalities for dynamic event embeddings."""

        if self.event_encoders is None:
            return []
        return [max(mapping.values(), default=0) + 1 for mapping in self.event_encoders.values()]


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


def fit_event_encoders(calendar: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Fit integer encoders for temporal event metadata columns."""

    encoders: dict[str, dict[str, int]] = {}
    for column in EVENT_COLUMNS:
        # Filter out NaN and get unique values
        values = sorted(calendar[column].dropna().unique().tolist())
        # 0 is reserved for "No Event"
        encoders[column] = {"__NONE__": 0, **{value: idx + 1 for idx, value in enumerate(values)}}
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


def _common_calendar_covariates(
    calendar: pd.DataFrame, 
    event_encoders: dict[str, dict[str, int]] | None = None
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Build calendar covariates including normalized time and event features.

    Returns:
        A tuple of (continuous_covs, continuous_names, categorical_covs, categorical_names).
    """

    # Extract date components for normalization
    dates = pd.to_datetime(calendar["date"])
    day = dates.dt.day.astype(float).to_numpy()
    week = dates.dt.isocalendar().week.astype(float).to_numpy()
    
    wday = calendar["wday"].astype(float).to_numpy()
    month = calendar["month"].astype(float).to_numpy()
    year = calendar["year"].astype(float).to_numpy()
    
    # M5 wday: 1 (Sat) to 7 (Fri)
    is_weekend = (wday <= 2).astype(float)

    # Normalization to [-0.5, 0.5]
    wday_norm = (wday - 4.0) / 6.0
    month_norm = (month - 6.5) / 11.0
    year_norm = (year - 2013.5) / 5.0
    day_norm = (day - 16.0) / 30.0
    week_norm = (week - 27.0) / 52.0

    cont_list = [
        np.sin(2.0 * np.pi * wday / 7.0),
        np.cos(2.0 * np.pi * wday / 7.0),
        np.sin(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * month / 12.0),
        wday_norm,
        month_norm,
        year_norm,
        day_norm,
        week_norm,
        is_weekend,
    ]
    cont_names = [
        "wday_sin", "wday_cos", "month_sin", "month_cos",
        "wday_norm", "month_norm", "year_norm", "day_norm", "week_norm",
        "is_weekend"
    ]

    cat_list = []
    cat_names = []

    if event_encoders is not None:
        for col in EVENT_COLUMNS:
            mapping = event_encoders[col]
            encoded = calendar[col].fillna("__NONE__").map(mapping).fillna(0).astype(np.float32).to_numpy()
            cat_list.append(encoded)
            cat_names.append(f"{col}_id")
    else:
        # Fallback to zeros/placeholders if no encoders
        for col in EVENT_COLUMNS:
            cat_list.append(np.zeros_like(wday))
            cat_names.append(f"{col}_id")

    continuous_covariates = np.stack(cont_list, axis=1).astype(np.float32)
    categorical_covariates = np.stack(cat_list, axis=1).astype(np.float32)
    
    return continuous_covariates, cont_names, categorical_covariates, cat_names


def _build_price_covariates(
    sales_frame: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Align weekly sell prices and compute relative price features.

    Returns:
        A tuple of (covariate_array, covariate_names).
        Features: log1p(price), price_missing, relative_price_max, relative_price_dept.
    """

    logger.debug("Building price covariates for %s series", len(sales_frame))
    calendar_weeks = calendar["wm_yr_wk"].to_numpy()
    num_series = len(sales_frame)
    num_days = len(calendar)

    # 1. Pre-calculate department means per week
    # We need item_id -> dept_id mapping
    item_dept_map = sales_frame[["item_id", "dept_id"]].drop_duplicates()
    prices_with_dept = prices.merge(item_dept_map, on="item_id", how="inner")
    dept_week_means = prices_with_dept.groupby(["dept_id", "wm_yr_wk"])["sell_price"].mean().to_dict()

    # 2. Filter prices to only selected series
    selected_keys = sales_frame[["store_id", "item_id"]].drop_duplicates()
    prices = prices.merge(selected_keys, on=["store_id", "item_id"], how="inner")
    
    price_groups = {
        key: group[["wm_yr_wk", "sell_price"]].to_numpy()
        for key, group in prices.groupby(["store_id", "item_id"], sort=False)
    }

    log_prices = np.zeros((num_series, num_days), dtype=np.float32)
    missing = np.ones_like(log_prices, dtype=np.float32)
    rel_max = np.zeros_like(log_prices, dtype=np.float32)
    rel_dept = np.zeros_like(log_prices, dtype=np.float32)

    for row_idx, row in enumerate(sales_frame.itertuples(index=False)):
        key = (row.store_id, row.item_id)
        dept = row.dept_id
        weekly_rows = price_groups.get(key)
        
        if weekly_rows is None or len(weekly_rows) == 0:
            continue
            
        weekly_map = {int(week): float(price) for week, price in weekly_rows}
        daily = np.array([weekly_map.get(int(week), np.nan) for week in calendar_weeks], dtype=np.float32)
        
        valid = np.isfinite(daily)
        missing[row_idx] = (~valid).astype(np.float32)
        
        fill_value = float(np.nanmedian(daily)) if valid.any() else 0.0
        daily = np.where(valid, daily, fill_value)
        
        # log1p price
        log_prices[row_idx] = np.log1p(np.maximum(daily, 0.0))
        
        # relative to max
        m_val = daily.max()
        if m_val > 0:
            rel_max[row_idx] = (daily / m_val) - 0.5 # Center roughly
            
        # relative to department mean
        dept_means = np.array([dept_week_means.get((dept, int(week)), daily_p) for week, daily_p in zip(calendar_weeks, daily)], dtype=np.float32)
        valid_dept = dept_means > 0
        rel_dept[row_idx] = np.where(valid_dept, (daily / dept_means) - 1.0, 0.0)

    covs = np.stack([log_prices, missing, rel_max, rel_dept], axis=2)
    names = ["log_sell_price", "price_missing", "relative_price_max", "relative_price_dept"]
    
    logger.info("Built price covariate cube shape=%s", covs.shape)
    return covs, names


def _calculate_zero_counts(values: np.ndarray) -> np.ndarray:
    """Pre-calculate continuous zero-sale days for each series and day.
    
    Result is a matrix of same shape as values, where each element is 
    the number of consecutive zero sales up to (but not including) that day.
    """
    
    num_series, num_days = values.shape
    zero_counts = np.zeros((num_series, num_days + 1), dtype=np.float32)
    
    for i in range(num_series):
        count = 0.0
        for j in range(num_days):
            zero_counts[i, j] = count
            if values[i, j] == 0:
                count += 1.0
            else:
                count = 0.0
        zero_counts[i, num_days] = count
        
    return zero_counts


def _build_covariate_cube(
    sales_frame: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    event_encoders: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Create the full ``series x day x feature`` known-covariate tensor.

    Features are strictly ordered: all Continuous first, then all Categorical.
    """

    num_series = len(sales_frame)
    num_days = len(calendar)
    
    # 1. Base components
    cont_common, cont_names, cat_common, cat_names = _common_calendar_covariates(calendar, event_encoders)
    price_covs, price_names = _build_price_covariates(sales_frame, calendar, prices)
    
    num_cont_common = cont_common.shape[1]
    num_price = price_covs.shape[2]
    num_cat = cat_common.shape[1]
    
    # 2. Re-ordered allocation: [Cont Common (10), SNAP (1), Price (4), Categorical (2)]
    # We leave Categorical for the very end of the cube
    covariate_names = cont_names + ["snap_state"] + price_names + cat_names
    covariates = np.zeros((num_series, num_days, len(covariate_names)), dtype=np.float32)
    
    logger.debug("Allocated covariate cube: %s", covariates.shape)

    # Fill continuous components
    covariates[:, :, :num_cont_common] = cont_common[None, :, :]
    
    # SNAP at index num_cont_common
    snap_idx = num_cont_common
    snap_map = {}
    for state in ["CA", "TX", "WI"]:
        snap_map[state] = calendar[f"snap_{state}"].astype(np.float32).to_numpy()
    
    state_ids = sales_frame["state_id"].astype(str).to_numpy()
    for state, snap_arr in snap_map.items():
        mask = (state_ids == state)
        if mask.any():
            covariates[mask, :, snap_idx] = snap_arr[None, :]

    # Price features start after SNAP
    price_start = snap_idx + 1
    covariates[:, :, price_start : price_start + num_price] = price_covs

    # Categorical features at the very end
    cat_start = price_start + num_price
    covariates[:, :, cat_start : cat_start + num_cat] = cat_common[None, :, :]

    logger.info("Built covariate cube shape=%s with features=%s", covariates.shape, covariate_names)
    return covariates, covariate_names


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


def _calculate_lags(values: np.ndarray, scales: np.ndarray, num_calendar_days: int) -> tuple[np.ndarray, list[str]]:
    """Compute scaled lag-28 features for each series.
    
    Lag-28 is a known covariate for the 28-day M5 forecast horizon.
    Lag-7 and rolling features must be calculated dynamically during the 
    autoregressive rollout to use predicted values.
    """

    num_series, num_days = values.shape
    scales_expanded = np.maximum(scales.reshape(-1, 1), 1e-4)
    scaled_values = values / scales_expanded

    lags = np.zeros((num_series, num_calendar_days, 1), dtype=np.float32)

    # Lag 28: available for t in [28, num_days + 27]
    max_t_lag28 = min(num_calendar_days, num_days + 28)
    lags[:, 28:max_t_lag28, 0] = scaled_values[:, : max_t_lag28 - 28]

    return lags, ["lag_28"]


def load_m5_bundle(
    config: DataConfig,
    encoders: dict[str, dict[str, int]] | None = None,
    event_encoders: dict[str, dict[str, int]] | None = None,
    series_ids: list[str] | None = None,
    load_covariates: bool = True,
) -> M5Bundle:
    """Load M5 CSVs and return model-ready arrays plus metadata."""

    data_dir = Path(config.data_dir)
    sales_path = data_dir / config.sales_file
    if not sales_path.exists():
        sales_path = data_dir / "sales_train_validation.csv"

    logger.info("Loading sales data from %s", sales_path)
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
        full_sales = pd.read_csv(sales_path, usecols=["id", *STATIC_COLUMNS])
        encoders = fit_encoders(full_sales)
        
    if event_encoders is None and config.use_event_embeddings:
        event_encoders = fit_event_encoders(calendar)

    sales_values = sales[day_columns].to_numpy(dtype=np.float32)
    train_end = max(1, sales_values.shape[1] - config.prediction_length)
    scales = _series_scales(sales_values, train_end=train_end)
    static_cats = encode_static(sales, encoders)
    
    # Pre-calculate zero counts (anchor states)
    zero_counts = _calculate_zero_counts(sales_values)

    if load_covariates:
        # 1. Base covariates strictly ordered: [Continuous (15), Categorical (2)]
        covariates, covariate_columns = _build_covariate_cube(sales, calendar, prices, event_encoders)
        num_cat = len(event_encoders) if event_encoders else 0
        
        # 2. Extract 28-day lags (Continuous)
        lag_covs, lag_names = _calculate_lags(sales_values, scales, calendar.shape[0])
        
        # 3. Final Concatenation strictly ordered: [Continuous (15), Lag (1), Categorical (2)]
        # This keeps all continuous features in a single block at the beginning
        cont_block = covariates[:, :, :-num_cat] if num_cat > 0 else covariates
        cat_block = covariates[:, :, -num_cat:] if num_cat > 0 else np.zeros((bundle.num_series, calendar.shape[0], 0))
        
        final_covs = np.concatenate([cont_block, lag_covs, cat_block], axis=2)
        final_names = covariate_columns[:-num_cat] + lag_names + covariate_columns[-num_cat:] if num_cat > 0 else covariate_columns + lag_names
        
        covariates = final_covs
        covariate_columns = final_names
    else:
        covariates = np.zeros((0, 0, 0), dtype=np.float32)
        covariate_columns = []

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
        event_encoders=event_encoders,
        zero_counts=zero_counts,
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

        # Extract 28 days of history before the window starts (prior_history)
        history_len = 28
        prior_history = np.zeros((len(series_idx), history_len), dtype=np.float32)
        for i, idx in enumerate(series_idx):
            h_start = max(0, start - history_len)
            h_end = start
            h_data = self.bundle.sales_values[idx, h_start:h_end]
            prior_history[i, history_len - len(h_data) :] = h_data

        initial_zero_counter = np.zeros((len(series_idx),), dtype=np.float32)
        if self.bundle.zero_counts is not None:
            initial_zero_counter = self.bundle.zero_counts[series_idx, start]

        batch = {
            "target": targets,
            "prior_history": prior_history,
            "initial_zero_counter": initial_zero_counter,
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

        # Extract 28 days of history before the window starts (prior_history)
        history_len = 28
        prior_history = np.zeros((len(series_idx), history_len), dtype=np.float32)
        for i, (s_idx, start) in enumerate(zip(series_idx, starts)):
            h_start = max(0, start - history_len)
            h_end = start
            h_data = self.bundle.sales_values[s_idx, h_start:h_end]
            prior_history[i, history_len - len(h_data) :] = h_data

        initial_zero_counter = np.zeros((len(series_idx),), dtype=np.float32)
        if self.bundle.zero_counts is not None:
            initial_zero_counter = self.bundle.zero_counts[series_idx, starts]

        loss_mask = np.zeros_like(targets, dtype=np.float32)
        loss_mask[:, self.context_length :] = 1.0
        return {
            "target": targets.astype(np.float32),
            "prior_history": prior_history.astype(np.float32),
            "initial_zero_counter": initial_zero_counter.astype(np.float32),
            "covariates": self.bundle.covariates[series_idx[:, None], positions].astype(np.float32),
            "static_cats": self.bundle.static_cats[series_idx].astype(np.int64),
            "scale": self.bundle.scales[series_idx, None].astype(np.float32),
            "loss_mask": loss_mask,
            "series_idx": series_idx.astype(np.int64),
        }

    def _log_batch_shapes(self, name: str, batch: dict[str, np.ndarray]) -> None:
        """Log the tensor-like shapes in a sampled batch for debugging."""

        logger.debug(
            "%s batch shapes: target=%s prior_history=%s covariates=%s static_cats=%s scale=%s loss_mask=%s series_idx=%s",
            name,
            batch["target"].shape,
            batch.get("prior_history", np.array([])).shape,
            batch["covariates"].shape,
            batch["static_cats"].shape,
            batch["scale"].shape,
            batch["loss_mask"].shape,
            batch["series_idx"].shape,
        )


def config_to_dict(config: DataConfig) -> dict:
    """Convert a data config dataclass to a serializable dictionary."""

    return asdict(config)
