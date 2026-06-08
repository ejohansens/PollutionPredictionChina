from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from air_quality.pipeline import (
    FORECAST_HORIZONS,
    make_numeric_feature_list,
    prepare_forecasting_dataset,
    transform_features,
    fit_preprocessor,
)
import matplotlib.pyplot as plt
import seaborn as sns


def _build_time_splits(dates: pd.Series, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.sort(dates.dt.to_period("h").drop_duplicates().dt.to_timestamp().values)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, test_index in tscv.split(unique_dates):
        train_dates = unique_dates[train_index]
        test_dates = unique_dates[test_index]
        splits.append((train_dates, test_dates))
    return splits


def _build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _fit_and_evaluate_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_features: List[str],
) -> Tuple[XGBRegressor, Dict[str, float]]:
    imputer, scaler = fit_preprocessor(X_train, numeric_features)
    X_train_prepared = transform_features(X_train, numeric_features, imputer, scaler)
    X_val_prepared = transform_features(X_val, numeric_features, imputer, scaler)

    model = _build_model()
    model.fit(X_train_prepared, y_train)

    y_pred = model.predict(X_val_prepared)
    metrics = _evaluate_predictions(y_val.to_numpy(), y_pred)
    return model, metrics


def evaluate_xgboost_walk_forward(
    df: pd.DataFrame,
    output_root: Path,
    horizons: List[int] = FORECAST_HORIZONS,
    n_splits: int = 3,
) -> pd.DataFrame:
    report_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    for horizon in horizons:
        print(f"Preparing horizon {horizon}h dataset...")
        X, y, feature_columns = prepare_forecasting_dataset(df, horizon)
        numeric_features = make_numeric_feature_list(feature_columns)
        timestamps = df.loc[X.index, "date"]

        splits = _build_time_splits(timestamps, n_splits=n_splits)

        for fold_index, (train_dates, val_dates) in enumerate(splits, start=1):
            train_mask = timestamps.isin(train_dates)
            val_mask = timestamps.isin(val_dates)

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_val = X.loc[val_mask]
            y_val = y.loc[val_mask]

            if X_train.empty or X_val.empty:
                continue

            print(f"Training horizon {horizon}h fold {fold_index}...")
            model, metrics = _fit_and_evaluate_fold(X_train, y_train, X_val, y_val, numeric_features)

            report_rows.append(
                {
                    "horizon_hours": horizon,
                    "fold": fold_index,
                    "train_rows": len(X_train),
                    "val_rows": len(X_val),
                    "train_start": train_dates.min(),
                    "train_end": train_dates.max(),
                    "val_start": val_dates.min(),
                    "val_end": val_dates.max(),
                    **metrics,
                }
            )

        print(f"Training final xgboost model for horizon {horizon}h on all data...")
        full_imputer, full_scaler = fit_preprocessor(X, numeric_features)
        X_all_prepared = transform_features(X, numeric_features, full_imputer, full_scaler)
        full_model = _build_model()
        full_model.fit(X_all_prepared, y)

        importance = pd.Series(full_model.feature_importances_, index=X.columns)
        for feature_name, importance_value in importance.sort_values(ascending=False).items():
            importance_rows.append(
                {
                    "horizon_hours": horizon,
                    "feature": feature_name,
                    "importance": float(importance_value),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(report_rows)
    importance_df = pd.DataFrame(importance_rows)
    report_df.to_csv(output_root / "xgboost_walk_forward_results.csv", index=False)
    importance_df.to_csv(output_root / "xgboost_feature_importance.csv", index=False)

    summary_lines = [
        f"XGBoost walk-forward evaluation saved to: {output_root / 'xgboost_walk_forward_results.csv'}",
        f"Feature importance saved to: {output_root / 'xgboost_feature_importance.csv'}",
        "",
        "Review the walk-forward metrics for MAE and RMSE across forecast horizons.",
    ]
    (output_root / "xgboost_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return report_df


def summarize_cv_report(report_df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Compute mean and std of MAE/RMSE across folds per horizon and save results and a plot."""
    summary = (
        report_df.groupby("horizon_hours")[["mae", "rmse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    # flatten columns
    summary.columns = [
        "horizon_hours",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "xgboost_cv_summary.csv", index=False)

    # Plot mean MAE with error bars using matplotlib
    plt.figure(figsize=(8, 4))
    x = summary["horizon_hours"].astype(str)
    y = summary["mae_mean"]
    yerr = summary["mae_std"].fillna(0)
    plt.bar(x, y, color="C0")
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="k", capsize=5)
    plt.title("XGBoost CV: MAE mean ± std by horizon")
    plt.xlabel("Horizon (hours)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_root / "cv_mae_bar.png")
    plt.close()

    return summary


def evaluate_time_holdout(
    df: pd.DataFrame,
    output_root: Path,
    holdout_days: int = 365,
    horizons: List[int] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    """Train on data before `holdout_days` and evaluate on the final `holdout_days` of data.

    Saves results to `xgboost_holdout_results.csv` and returns the DataFrame.
    """
    holdout_rows: List[Dict[str, object]] = []

    for horizon in horizons:
        X, y, feature_columns = prepare_forecasting_dataset(df, horizon)
        numeric_features = make_numeric_feature_list(feature_columns)
        timestamps = df.loc[X.index, "date"]

        if timestamps.empty:
            continue

        holdout_start = timestamps.max() - pd.Timedelta(days=holdout_days)
        train_mask = timestamps < holdout_start
        val_mask = timestamps >= holdout_start

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_val = X.loc[val_mask]
        y_val = y.loc[val_mask]

        if X_train.empty or X_val.empty:
            continue

        imputer, scaler = fit_preprocessor(X_train, numeric_features)
        X_train_prepared = transform_features(X_train, numeric_features, imputer, scaler)
        X_val_prepared = transform_features(X_val, numeric_features, imputer, scaler)

        model = _build_model()
        model.fit(X_train_prepared, y_train)
        y_pred = model.predict(X_val_prepared)
        metrics = _evaluate_predictions(y_val.to_numpy(), y_pred)

        holdout_rows.append(
            {
                "horizon_hours": horizon,
                "train_rows": len(X_train),
                "holdout_rows": len(X_val),
                "train_start": timestamps.loc[train_mask].min(),
                "train_end": timestamps.loc[train_mask].max(),
                "holdout_start": timestamps.loc[val_mask].min(),
                "holdout_end": timestamps.loc[val_mask].max(),
                **metrics,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(output_root / "xgboost_holdout_results.csv", index=False)

    return holdout_df


def plot_cv_and_holdout(report_df: pd.DataFrame, holdout_df: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    # Boxplot of MAE per horizon across CV folds
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="horizon_hours", y="mae", data=report_df)
    plt.title("CV MAE distribution by horizon")
    plt.xlabel("Horizon (hours)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_root / "cv_mae_boxplot.png")
    plt.close()

    # Holdout vs CV mean comparison
    cv_summary = summarize_cv_report(report_df, output_root)
    if not holdout_df.empty:
        merged = cv_summary.merge(holdout_df[["horizon_hours", "mae", "rmse"]], on="horizon_hours", how="left", suffixes=("_cv", "_holdout"))
        plt.figure(figsize=(8, 4))
        x = merged["horizon_hours"].astype(str)
        plt.plot(x, merged["mae_mean"], marker="o", label="CV mean MAE")
        plt.plot(x, merged["mae"], marker="o", label="Holdout MAE")
        plt.title("CV mean MAE vs Holdout MAE")
        plt.xlabel("Horizon (hours)")
        plt.ylabel("MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_root / "cv_vs_holdout_mae.png")
        plt.close()
