from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

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


def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
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


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature in LAG_FEATURE_COLUMNS:
        for lag in LAG_HORIZONS:
            df[f"{feature}_lag{lag}"] = df.groupby("station")[feature].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature in ["PM2.5", "PM10"]:
        for window in ROLLING_WINDOWS:
            df[f"{feature}_roll{window}"] = (
                df.groupby("station")[feature]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df


def _encode_station_dummies(df: pd.DataFrame) -> pd.DataFrame:
    station_dummies = pd.get_dummies(df["station"], prefix="station", dtype=int)
    return pd.concat([df, station_dummies], axis=1)


def _get_feature_columns(df: pd.DataFrame, target_column: str) -> List[str]:
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


def prepare_forecasting_dataset(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.sort_values(["station", "date"]).copy()
    df = _add_cyclical_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _encode_station_dummies(df)

    future_target = f"{TARGET_COLUMN}_future_{horizon}"
    df[future_target] = df.groupby("station")[TARGET_COLUMN].shift(-horizon)

    lag_and_roll_columns = [
        f"{feature}_lag{lag}"
        for feature in LAG_FEATURE_COLUMNS
        for lag in LAG_HORIZONS
    ] + [
        f"{feature}_roll{window}"
        for feature in ["PM2.5", "PM10"]
        for window in ROLLING_WINDOWS
    ]

    # Keep the original DataFrame indices so downstream code can map rows
    # back to the original `date` values when building time-based splits.
    df = df.dropna(subset=lag_and_roll_columns + [future_target])

    feature_columns = _get_feature_columns(df, future_target)
    X = df[feature_columns].copy()
    y = df[future_target].copy()
    return X, y, feature_columns


def fit_preprocessor(X: pd.DataFrame, numeric_features: List[str]) -> Tuple[KNNImputer, StandardScaler]:
    imputer = KNNImputer(n_neighbors=5)
    X_numeric = imputer.fit_transform(X[numeric_features])
    scaler = StandardScaler()
    scaler.fit(X_numeric)
    return imputer, scaler


def transform_features(
    X: pd.DataFrame,
    numeric_features: List[str],
    imputer: KNNImputer,
    scaler: StandardScaler,
) -> pd.DataFrame:
    result = X.copy()
    result[numeric_features] = scaler.transform(imputer.transform(result[numeric_features]))
    return result


def make_numeric_feature_list(feature_columns: List[str]) -> List[str]:
    return [col for col in feature_columns if not col.startswith("station_")]
