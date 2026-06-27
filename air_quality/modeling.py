from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from air_quality.pipeline import (
    FORECAST_HORIZONS,
    MODELS,
    make_numeric_feature_list,
    prepare_forecasting_dataset,
    transform_features,
    fit_preprocessor,
)
import matplotlib.pyplot as plt
import seaborn as sns


class LinearTreeHybrid(BaseEstimator, RegressorMixin):
    """Two-stage hybrid regressor.

    Stage 1 — Ridge regression extracts the linear component of the signal.
    Stage 2 — High-capacity LightGBM (1000 trees, small lr, wide leaves) fits
              only the residuals left by Ridge, modelling non-linear structure.
    Final prediction = linear_stage + residual_stage.

    Feature importances are taken from the residual LightGBM, showing which
    features drive the part of the signal that cannot be captured linearly.
    """

    def __init__(
        self,
        ridge_alpha: float = 1.0,
        n_estimators: int = 1000,
        learning_rate: float = 0.02,
        num_leaves: int = 63,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def fit(self, X, y):
        self.linear_model_ = Ridge(alpha=self.ridge_alpha)
        self.linear_model_.fit(X, y)
        residuals = y - self.linear_model_.predict(X)

        self.residual_model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.residual_model_.fit(X, residuals)
        return self

    def predict(self, X):
        return self.linear_model_.predict(X) + self.residual_model_.predict(X)

    @property
    def feature_importances_(self):
        """Importances from the residual LightGBM stage."""
        return self.residual_model_.feature_importances_


def _build_time_splits(dates: pd.Series, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.sort(dates.dt.to_period("h").drop_duplicates().dt.to_timestamp().values)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, test_index in tscv.split(unique_dates):
        train_dates = unique_dates[train_index]
        test_dates = unique_dates[test_index]
        splits.append((train_dates, test_dates))
    return splits


def _build_xgboost_model() -> XGBRegressor:
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

def _build_lgbm_model() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )


def _build_catboost_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    )


def _build_ridge_model() -> Ridge:
    """Ridge regression — linear baseline with L2 regularisation."""
    return Ridge(alpha=1.0)


def _build_elasticnet_model() -> ElasticNet:
    """ElasticNet — L1 + L2 regularisation.
    The L1 term zeros out irrelevant coefficients (automatic sparsity);
    L2 stabilises groups of correlated features.  Complements Ridge by
    showing how much of the signal is truly linear and sparse."""
    return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42)


def _build_stacking_model() -> StackingRegressor:
    """Stacking ensemble: XGBoost, LightGBM, and CatBoost as base learners;
    Ridge regression as the meta-learner.
    cv=3 uses KFold internally — required because cross_val_predict (used by
    StackingRegressor) needs every sample to appear in exactly one test fold,
    which TimeSeriesSplit cannot guarantee.  The outer walk-forward CV already
    enforces time ordering, so this is the standard pragmatic choice."""
    estimators = [
        ("xgboost",  _build_xgboost_model()),
        ("lightgbm", _build_lgbm_model()),
        ("catboost", _build_catboost_model()),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=1,          # inner models already parallelise internally
        passthrough=False, # meta-learner sees base-model predictions only
    )

def _build_hybrid_model() -> LinearTreeHybrid:
    """Linear-tree hybrid: Ridge stage + high-capacity LightGBM residual stage."""
    return LinearTreeHybrid(
        ridge_alpha=1.0,
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=63,
    )


def _build_model(model_name: str):
    if model_name == "xgboost":
        return _build_xgboost_model()
    elif model_name == "lightgbm":
        return _build_lgbm_model()
    elif model_name == "catboost":
        return _build_catboost_model()
    elif model_name == "ridge":
        return _build_ridge_model()
    elif model_name == "elasticnet":
        return _build_elasticnet_model()
    elif model_name == "hybrid":
        return _build_hybrid_model()
    elif model_name == "stacking":
        return _build_stacking_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _fit_and_evaluate_fold(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_features: List[str],
) -> Tuple[XGBRegressor, Dict[str, float]]:
    
    imputer, scaler = fit_preprocessor(X_train, numeric_features)
    X_train_prepared = transform_features(X_train, numeric_features, imputer, scaler)
    X_val_prepared = transform_features(X_val, numeric_features, imputer, scaler)

    model.fit(X_train_prepared, y_train)

    y_pred = model.predict(X_val_prepared)
    metrics = _evaluate_predictions(y_val.to_numpy(), y_pred)
    return model, metrics


def evaluate_models_walk_forward(
    df: pd.DataFrame,
    output_root: Path,
    models: List[str] = MODELS,
    horizons: List[int] = FORECAST_HORIZONS,
    n_splits: int = 3,
) -> Dict[str, pd.DataFrame]:

    all_results = {}

    for model_name in models:

        print(f"\n==============================")
        print(f"Evaluating model: {model_name}")
        print(f"==============================\n")

        report_rows = []
        importance_rows = []

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

                print(f"[{model_name}] horizon {horizon}h fold {fold_index}")

                model = _build_model(model_name)

                model, metrics = _fit_and_evaluate_fold(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    numeric_features,
                )

                report_rows.append(
                    {
                        "model": model_name,
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

            # Train a final model on ALL data for this horizon to get importances.
            # Each horizon has a different feature set, so this must be done per horizon.
            print(f"  Training importance model: {model_name} @ {horizon}h")
            full_X, full_y, feat_cols = prepare_forecasting_dataset(df, horizon)
            nf = make_numeric_feature_list(feat_cols)
            imp_model = _build_model(model_name)
            imp_imputer, imp_scaler = fit_preprocessor(full_X, nf)
            X_all = transform_features(full_X, nf, imp_imputer, imp_scaler)
            imp_model.fit(X_all, full_y)

            if hasattr(imp_model, "feature_importances_"):
                imp_series = pd.Series(imp_model.feature_importances_, index=X_all.columns)
            elif isinstance(imp_model, (Ridge, ElasticNet)):
                imp_series = pd.Series(np.abs(imp_model.coef_), index=X_all.columns)
            elif isinstance(imp_model, StackingRegressor):
                base_names = [name for name, _ in imp_model.estimators]
                imp_series = pd.Series(np.abs(imp_model.final_estimator_.coef_), index=base_names)
            else:
                imp_series = pd.Series(dtype=float)

            for feat, val in imp_series.sort_values(ascending=False).items():
                importance_rows.append(
                    {
                        "model": model_name,
                        "horizon_hours": horizon,
                        "feature": feat,
                        "importance": float(val),
                    }
                )

        output_root.mkdir(parents=True, exist_ok=True)

        report_df = pd.DataFrame(report_rows)
        importance_df = pd.DataFrame(importance_rows)

        report_df.to_csv(output_root / f"{model_name}_walk_forward.csv", index=False)
        # One combined CSV (all horizons) + one CSV per horizon for easy inspection
        importance_df.to_csv(output_root / f"{model_name}_importance.csv", index=False)
        for h, grp in importance_df.groupby("horizon_hours"):
            grp.drop(columns="horizon_hours").to_csv(
                output_root / f"{model_name}_importance_{h}h.csv", index=False
            )

        all_results[model_name] = report_df

    return all_results


def summarize_cv_report(
    report_df: pd.DataFrame,
    output_root: Path,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Compute mean and std of MAE/RMSE across folds per horizon and save results and a plot."""
    if model_name is None:
        if "model" in report_df.columns and not report_df["model"].dropna().empty:
            model_name = str(report_df["model"].dropna().iloc[0])
        else:
            model_name = "model"

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
    summary.to_csv(output_root / f"{model_name}_cv_summary.csv", index=False)

    # Plot mean MAE with error bars using matplotlib
    plt.figure(figsize=(8, 4))
    x = summary["horizon_hours"].astype(str)
    y = summary["mae_mean"]
    yerr = summary["mae_std"].fillna(0)
    plt.bar(x, y, color="C0")
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="k", capsize=5)
    plt.title(f"{model_name.upper()} CV: MAE mean ± std by horizon")
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
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Train on data before `holdout_days` and evaluate on the final
    `holdout_days` of data.

    Saves results to `<model_name>_holdout_results.csv`.
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

        X_train_prepared = transform_features(
            X_train,
            numeric_features,
            imputer,
            scaler,
        )

        X_val_prepared = transform_features(
            X_val,
            numeric_features,
            imputer,
            scaler,
        )

        model = _build_model(model_name)

        model.fit(X_train_prepared, y_train)

        y_pred = model.predict(X_val_prepared)

        metrics = _evaluate_predictions(
            y_val.to_numpy(),
            y_pred,
        )

        holdout_rows.append(
            {
                "model": model_name,
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

    holdout_df.to_csv(
        output_root / f"{model_name}_holdout_results.csv",
        index=False,
    )

    return holdout_df


def compare_models(results: Dict[str, pd.DataFrame], output_root: Path):

    summary = []

    for model_name, df in results.items():

        grouped = df.groupby("horizon_hours")[["mae", "rmse"]].mean().reset_index()
        grouped["model"] = model_name

        summary.append(grouped)

    summary_df = pd.concat(summary)

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=summary_df,
        x="horizon_hours",
        y="mae",
        hue="model",
        marker="o"
    )

    plt.title("Model comparison (MAE vs horizon)")
    plt.tight_layout()

    output_root.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_root / "model_comparison_mae.png")
    plt.close()

    return summary_df

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
