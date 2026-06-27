from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

NUMERIC_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

LAG_FEATURE_COLUMNS = ["PM2.5", "PM10", "TEMP", "PRES", "DEWP", "WSPM"]
LAG_HORIZONS = [1, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
FORECAST_HORIZONS = [1, 3, 6, 12, 24]
TARGET_COLUMN = "PM2.5"
MODELS = ["xgboost", "lightgbm", "catboost", "ridge", "elasticnet", "hybrid", "stacking"]

# Horizon-aware feature config: lags and rolling windows relevant per forecast horizon.
# Short lags (1h, 3h) are dropped for longer horizons where they carry little signal.
HORIZON_LAG_CONFIG: dict = {
    1:  {"lags": [1, 3, 6],       "rolling": [3, 6]},
    3:  {"lags": [1, 3, 6, 12],   "rolling": [3, 6, 12]},
    6:  {"lags": [3, 6, 12],      "rolling": [6, 12]},
    12: {"lags": [6, 12, 24],     "rolling": [6, 12, 24]},
    24: {"lags": [12, 24],        "rolling": [12, 24]},
}
CORRELATION_THRESHOLD = 0.05
COLLINEARITY_THRESHOLD = 0.92  # drop one from each highly-collinear feature pair

# Compass direction -> degrees for wind direction encoding.
# The raw `wd` string column is excluded from features; derived sin/cos are included.
WD_DEGREES: dict = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour, month, and day-of-week as sin/cos pairs to preserve their circular nature."""
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def _add_lag_features(df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
    """Shift key sensor/met columns by each lag (hours) within each station group."""
    if lags is None:
        lags = LAG_HORIZONS
    df = df.copy()
    for feature in LAG_FEATURE_COLUMNS:
        for lag in lags:
            df[f"{feature}_lag{lag}"] = df.groupby("station")[feature].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """Rolling mean of PM2.5 and PM10 over each window (hours), shifted by 1 to avoid leakage."""
    if windows is None:
        windows = ROLLING_WINDOWS
    df = df.copy()
    for feature in ["PM2.5", "PM10"]:
        for window in windows:
            df[f"{feature}_roll{window}"] = (
                df.groupby("station")[feature]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df


def _add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode compass wind direction as sin/cos so the circular nature is preserved."""
    df = df.copy()
    if "wd" not in df.columns:
        return df
    degrees = df["wd"].map(WD_DEGREES)
    rad = np.deg2rad(degrees)
    df["wd_sin"] = np.sin(rad)   # NaN propagates for unmapped / missing directions
    df["wd_cos"] = np.cos(rad)
    return df


def _add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics- and domain-motivated derived features:
    - dew_depression  : DEWP - TEMP  (negative = dry air; near 0 = fog/high humidity)
    - pm_fine_ratio   : PM2.5 / (PM10 + 1)  (share of fine particles; combustion signal)
    - pm10_excess     : PM10 - PM2.5  (coarse particle fraction; dust / road signal)
    """
    df = df.copy()
    if "DEWP" in df.columns and "TEMP" in df.columns:
        df["dew_depression"] = df["DEWP"] - df["TEMP"]
    if "PM2.5" in df.columns and "PM10" in df.columns:
        df["pm_fine_ratio"] = df["PM2.5"] / (df["PM10"].abs() + 1.0)
        df["pm10_excess"] = df["PM10"] - df["PM2.5"]
    return df


def _add_diff_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Rate-of-change features for PM2.5: current_value - value_lag_hours_ago.
    Captures whether pollution is rising or falling. Only computed for lags
    that were actually generated (so the lag column already exists).
    """
    df = df.copy()
    for lag in lags:
        col = f"PM2.5_lag{lag}"
        if col in df.columns:
            df[f"PM2.5_diff{lag}"] = df["PM2.5"] - df[col]
    return df


def _add_rolling_std_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Rolling standard deviation for PM2.5 and PM10.
    Captures volatility / event-driven spikes that rolling mean misses.
    """
    df = df.copy()
    for feature in ["PM2.5", "PM10"]:
        for window in windows:
            df[f"{feature}_rollstd{window}"] = (
                df.groupby("station")[feature]
                .shift(1)
                .rolling(window=window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )
    return df


def _select_features_by_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = CORRELATION_THRESHOLD,
) -> List[str]:
    """Keep features whose absolute Pearson correlation with `y` meets the threshold.
    Station dummy columns are always kept regardless of correlation."""
    station_cols = [c for c in X.columns if c.startswith("station_")]
    numeric_cols = [c for c in X.columns if c not in station_cols]
    corr = X[numeric_cols].corrwith(y).abs()
    kept_numeric = corr[corr >= threshold].index.tolist()
    return kept_numeric + station_cols


def _drop_collinear_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_correlation: float = COLLINEARITY_THRESHOLD,
) -> List[str]:
    """Remove one feature from each highly-collinear pair.
    When |Pearson r| between two numeric features exceeds `max_correlation`,
    the one with lower absolute correlation to target `y` is dropped.
    Station dummy columns are never dropped.
    """
    station_cols = [c for c in X.columns if c.startswith("station_")]
    numeric_cols = [c for c in X.columns if c not in station_cols]

    if len(numeric_cols) < 2:
        return list(X.columns)

    corr_matrix = X[numeric_cols].corr().abs()
    target_corr = X[numeric_cols].corrwith(y).abs().fillna(0)

    to_drop: set = set()
    for i, col_i in enumerate(numeric_cols):
        if col_i in to_drop:
            continue
        for col_j in numeric_cols[i + 1 :]:
            if col_j in to_drop:
                continue
            if corr_matrix.loc[col_i, col_j] > max_correlation:
                # Keep the feature with stronger correlation to the target
                if target_corr[col_i] >= target_corr[col_j]:
                    to_drop.add(col_j)
                else:
                    to_drop.add(col_i)

    kept = [c for c in numeric_cols if c not in to_drop] + station_cols
    return kept


def _encode_station_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the station column and append the dummy columns to the DataFrame."""
    station_dummies = pd.get_dummies(df["station"], prefix="station", dtype=int)
    return pd.concat([df, station_dummies], axis=1)


def _get_feature_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    """Return all columns that are usable as model features (excludes date, raw categoricals, and the target)."""
    excluded_columns = {
        "date",
        "year",
        "month",
        "day",
        "hour",
        "dayofweek",
        "wd",
        "station",
        "source_file",
        target_column,
    }
    return [col for col in df.columns if col not in excluded_columns]


def prepare_forecasting_dataset(
    df: pd.DataFrame,
    horizon: int,
    min_correlation: float = CORRELATION_THRESHOLD,
    max_collinearity: float = COLLINEARITY_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Build X, y for a given forecast horizon with full feature engineering and two-stage filtering.
    Applies horizon-specific lags/windows, then drops low-correlation and collinear features.
    """
    df = df.sort_values(["station", "date"]).copy()
    df = _add_cyclical_time_features(df)
    df = _add_wind_direction_features(df)   # wd -> wd_sin, wd_cos
    df = _add_domain_features(df)           # dew_depression, pm_fine_ratio, pm10_excess

    # Use horizon-specific lag and rolling-window config to keep only
    # features that are likely to carry signal for this forecast distance.
    horizon_config = HORIZON_LAG_CONFIG.get(
        horizon, {"lags": LAG_HORIZONS, "rolling": ROLLING_WINDOWS}
    )
    lags = horizon_config["lags"]
    windows = horizon_config["rolling"]

    df = _add_lag_features(df, lags=lags)
    df = _add_rolling_features(df, windows=windows)
    df = _add_rolling_std_features(df, windows=windows)  # volatility signal
    df = _add_diff_features(df, lags=lags)               # rate-of-change / trend
    df = _encode_station_dummies(df)

    future_target = f"{TARGET_COLUMN}_future_{horizon}"
    df[future_target] = df.groupby("station")[TARGET_COLUMN].shift(-horizon)

    # Only require core lag/roll columns for the dropna so that optional
    # features with partial NaN coverage (e.g. rollstd with min_periods=2)
    # do not discard extra rows.
    core_columns = [
        f"{feature}_lag{lag}"
        for feature in LAG_FEATURE_COLUMNS
        for lag in lags
    ] + [
        f"{feature}_roll{window}"
        for feature in ["PM2.5", "PM10"]
        for window in windows
    ]

    # Keep the original DataFrame indices so downstream code can map rows
    # back to the original `date` values when building time-based splits.
    df = df.dropna(subset=core_columns + [future_target])

    feature_columns = _get_feature_columns(df, future_target)
    X = df[feature_columns].copy()
    y = df[future_target].copy()

    # Step 1: drop features with near-zero correlation to target.
    if min_correlation > 0:
        selected = _select_features_by_correlation(X, y, threshold=min_correlation)
        X = X[selected]

    # Step 2: from remaining features, drop one from each collinear pair.
    if max_collinearity < 1.0:
        selected = _drop_collinear_features(X, y, max_correlation=max_collinearity)
        X = X[selected]

    feature_columns = list(X.columns)
    return X, y, feature_columns


def fit_preprocessor(X: pd.DataFrame, numeric_features: List[str]) -> Tuple[SimpleImputer, RobustScaler]:
    """Fit median imputer and RobustScaler on training features.
    Median imputation is sufficient given < 0.5 % NaN rate; KNN imputation
    would be O(n²) on 300k-row training folds and add no meaningful benefit."""
    imputer = SimpleImputer(strategy="median")
    X_numeric = imputer.fit_transform(X[numeric_features])
    scaler = RobustScaler()
    scaler.fit(X_numeric)
    return imputer, scaler


def transform_features(
    X: pd.DataFrame,
    numeric_features: List[str],
    imputer: SimpleImputer,
    scaler: RobustScaler,
) -> pd.DataFrame:
    """Apply the fitted imputer and scaler to X (numeric columns only)."""
    result = X.copy()
    result[numeric_features] = scaler.transform(imputer.transform(result[numeric_features]))
    return result


def make_numeric_feature_list(feature_columns: List[str]) -> List[str]:
    """Return only the continuous feature names, excluding one-hot station dummies."""
    return [col for col in feature_columns if not col.startswith("station_")]
