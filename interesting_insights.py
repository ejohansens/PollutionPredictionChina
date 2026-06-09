from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


MODELS = ["xgboost", "lightgbm", "catboost"]


def _read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    print(f"[WARN] File not found: {path}")
    return None


def _save_bar(df: pd.DataFrame, x: str, y: str, title: str, output_path: Path, rotation: int = 0) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_line(df: pd.DataFrame, x: str, y: str, group: str, title: str, output_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    for name, group_df in df.groupby(group):
        group_df = group_df.sort_values(x)
        plt.plot(group_df[x], group_df[y], marker="o", label=str(name))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_walk_forward_results(benchmark_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model_name in MODELS:
        path = benchmark_root / f"{model_name}_walk_forward.csv"
        df = _read_csv_if_exists(path)
        if df is not None and not df.empty:
            df["model"] = model_name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_importances(benchmark_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model_name in MODELS:
        path = benchmark_root / f"{model_name}_importance.csv"
        df = _read_csv_if_exists(path)
        if df is not None and not df.empty:
            df["model"] = model_name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_holdout_results(benchmark_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model_name in MODELS:
        path = benchmark_root / model_name / f"{model_name}_holdout_results.csv"
        df = _read_csv_if_exists(path)
        if df is not None and not df.empty:
            df["model"] = model_name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_model_insights(walk_forward: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if walk_forward.empty:
        return {}

    cv_summary = (
        walk_forward
        .groupby(["model", "horizon_hours"])[["mae", "rmse"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    cv_summary.columns = [
        "model",
        "horizon_hours",
        "mae_mean",
        "mae_std",
        "mae_min",
        "mae_max",
        "rmse_mean",
        "rmse_std",
        "rmse_min",
        "rmse_max",
    ]
    cv_summary["mae_cv_percent"] = 100 * cv_summary["mae_std"] / cv_summary["mae_mean"]
    cv_summary["rmse_cv_percent"] = 100 * cv_summary["rmse_std"] / cv_summary["rmse_mean"]

    best_by_horizon = (
        cv_summary
        .sort_values(["horizon_hours", "mae_mean", "rmse_mean"])
        .groupby("horizon_hours", as_index=False)
        .first()
    )

    baseline = cv_summary[cv_summary["horizon_hours"] == cv_summary["horizon_hours"].min()][
        ["model", "mae_mean", "rmse_mean"]
    ].rename(columns={"mae_mean": "mae_at_shortest_horizon", "rmse_mean": "rmse_at_shortest_horizon"})

    longest = cv_summary[cv_summary["horizon_hours"] == cv_summary["horizon_hours"].max()][
        ["model", "mae_mean", "rmse_mean"]
    ].rename(columns={"mae_mean": "mae_at_longest_horizon", "rmse_mean": "rmse_at_longest_horizon"})

    horizon_degradation = baseline.merge(longest, on="model", how="inner")
    horizon_degradation["mae_delta"] = (
        horizon_degradation["mae_at_longest_horizon"]
        - horizon_degradation["mae_at_shortest_horizon"]
    )
    horizon_degradation["mae_ratio_long_vs_short"] = (
        horizon_degradation["mae_at_longest_horizon"]
        / horizon_degradation["mae_at_shortest_horizon"]
    )
    horizon_degradation = horizon_degradation.sort_values("mae_ratio_long_vs_short")

    return {
        "cv_summary_by_model_horizon": cv_summary,
        "best_model_by_horizon": best_by_horizon,
        "horizon_degradation": horizon_degradation,
    }


def build_feature_insights(importances: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if importances.empty:
        return {}

    normalized = importances.copy()
    normalized["importance_share"] = normalized.groupby("model")["importance"].transform(
        lambda s: s / s.sum() if s.sum() else s
    )

    top_features_by_model = (
        normalized
        .sort_values(["model", "importance"], ascending=[True, False])
        .groupby("model")
        .head(20)
    )

    feature_consensus = (
        normalized
        .groupby("feature")
        .agg(
            mean_importance_share=("importance_share", "mean"),
            max_importance_share=("importance_share", "max"),
            models_present=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["models_present", "mean_importance_share"], ascending=[False, False])
    )

    return {
        "top_features_by_model": top_features_by_model,
        "feature_consensus": feature_consensus,
    }


def build_data_quality_insights(output_root: Path) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}

    station_summary = _read_csv_if_exists(output_root / "station_summary.csv")
    if station_summary is not None and not station_summary.empty:
        if "station" not in station_summary.columns:
            station_summary = station_summary.rename(columns={station_summary.columns[0]: "station"})
        results["most_polluted_stations_pm25"] = station_summary.sort_values("mean_PM2.5", ascending=False).head(10)
        results["cleanest_stations_pm25"] = station_summary.sort_values("mean_PM2.5", ascending=True).head(10)
        if "missing_rows" in station_summary.columns and "rows" in station_summary.columns:
            station_summary["missing_row_percent"] = 100 * station_summary["missing_rows"] / station_summary["rows"]
            results["stations_with_most_missing_rows"] = station_summary.sort_values(
                "missing_row_percent", ascending=False
            ).head(10)

    missing_report = _read_csv_if_exists(output_root / "missing_report.csv")
    if missing_report is not None and not missing_report.empty:
        if "variable" not in missing_report.columns:
            missing_report = missing_report.rename(columns={missing_report.columns[0]: "variable"})
        results["variables_with_most_missing_values"] = missing_report.sort_values(
            "missing_percent", ascending=False
        ).head(10)

    monthly_summary = _read_csv_if_exists(output_root / "monthly_summary.csv")
    if monthly_summary is not None and not monthly_summary.empty:
        monthly_summary["month"] = pd.to_datetime(monthly_summary["year_month"]).dt.month
        seasonal_pm25 = (
            monthly_summary
            .groupby("month")["mean_PM2.5"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .rename(columns={"mean": "mean_PM2.5", "min": "min_PM2.5", "max": "max_PM2.5"})
            .sort_values("mean_PM2.5", ascending=False)
        )
        results["seasonal_pm25_by_month"] = seasonal_pm25

    return results


def build_holdout_insights(walk_forward: pd.DataFrame, holdout: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if walk_forward.empty or holdout.empty:
        return {}

    cv_mean = (
        walk_forward
        .groupby(["model", "horizon_hours"])[["mae", "rmse"]]
        .mean()
        .reset_index()
        .rename(columns={"mae": "cv_mae_mean", "rmse": "cv_rmse_mean"})
    )
    holdout_small = holdout[["model", "horizon_hours", "mae", "rmse"]].rename(
        columns={"mae": "holdout_mae", "rmse": "holdout_rmse"}
    )
    comparison = cv_mean.merge(holdout_small, on=["model", "horizon_hours"], how="inner")
    comparison["holdout_mae_minus_cv"] = comparison["holdout_mae"] - comparison["cv_mae_mean"]
    comparison["holdout_mae_ratio_vs_cv"] = comparison["holdout_mae"] / comparison["cv_mae_mean"]
    comparison = comparison.sort_values(["horizon_hours", "holdout_mae_ratio_vs_cv"])
    return {"cv_vs_holdout": comparison}


def save_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def write_markdown_report(tables: Dict[str, pd.DataFrame], output_path: Path) -> None:
    lines = [
        "# Interesting insights from existing outputs",
        "",
        "This report is generated only from files already saved in `output/`. It does not retrain models.",
        "",
    ]

    if "best_model_by_horizon" in tables and not tables["best_model_by_horizon"].empty:
        lines += ["## Best model by forecast horizon", ""]
        for _, row in tables["best_model_by_horizon"].iterrows():
            lines.append(
                f"- {int(row['horizon_hours'])}h: {row['model']} "
                f"with MAE={row['mae_mean']:.3f} and RMSE={row['rmse_mean']:.3f}."
            )
        lines.append("")

    if "horizon_degradation" in tables and not tables["horizon_degradation"].empty:
        lines += ["## Forecast degradation from shortest to longest horizon", ""]
        for _, row in tables["horizon_degradation"].iterrows():
            lines.append(
                f"- {row['model']}: MAE ratio longest/shortest = "
                f"{row['mae_ratio_long_vs_short']:.2f} "
                f"(delta={row['mae_delta']:.3f})."
            )
        lines.append("")

    if "feature_consensus" in tables and not tables["feature_consensus"].empty:
        lines += ["## Features with strongest consensus across models", ""]
        top = tables["feature_consensus"].head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['feature']}: average normalized importance "
                f"{100 * row['mean_importance_share']:.2f}% across {int(row['models_present'])} model(s)."
            )
        lines.append("")

    if "most_polluted_stations_pm25" in tables and not tables["most_polluted_stations_pm25"].empty:
        lines += ["## Stations with highest average PM2.5", ""]
        for _, row in tables["most_polluted_stations_pm25"].head(5).iterrows():
            lines.append(f"- {row['station']}: mean PM2.5 = {row['mean_PM2.5']:.2f}.")
        lines.append("")

    if "seasonal_pm25_by_month" in tables and not tables["seasonal_pm25_by_month"].empty:
        top_month = tables["seasonal_pm25_by_month"].iloc[0]
        lines += [
            "## Seasonal PM2.5 pattern",
            "",
            f"- Highest average PM2.5 month: {int(top_month['month'])}, "
            f"mean PM2.5 = {top_month['mean_PM2.5']:.2f}.",
            "",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    output_root = workspace_root / "output"
    benchmark_root = output_root / "benchmark"
    insights_root = output_root / "insights"
    insights_root.mkdir(parents=True, exist_ok=True)

    walk_forward = load_walk_forward_results(benchmark_root)
    importances = load_importances(benchmark_root)
    holdout = load_holdout_results(benchmark_root)

    tables: Dict[str, pd.DataFrame] = {}
    tables.update(build_model_insights(walk_forward))
    tables.update(build_feature_insights(importances))
    tables.update(build_data_quality_insights(output_root))
    tables.update(build_holdout_insights(walk_forward, holdout))

    save_tables(tables, insights_root)

    if "cv_summary_by_model_horizon" in tables:
        _save_line(
            tables["cv_summary_by_model_horizon"],
            x="horizon_hours",
            y="mae_mean",
            group="model",
            title="Mean CV MAE by model and forecast horizon",
            output_path=insights_root / "cv_mae_by_model_horizon.png",
        )

    if "feature_consensus" in tables:
        _save_bar(
            tables["feature_consensus"].head(20),
            x="feature",
            y="mean_importance_share",
            title="Top 20 features by average normalized importance",
            output_path=insights_root / "top_feature_consensus.png",
            rotation=75,
        )

    if "seasonal_pm25_by_month" in tables:
        _save_bar(
            tables["seasonal_pm25_by_month"].sort_values("month"),
            x="month",
            y="mean_PM2.5",
            title="Average PM2.5 by calendar month",
            output_path=insights_root / "seasonal_pm25_by_month.png",
        )

    write_markdown_report(tables, insights_root / "insights_report.md")

    print(f"Insights saved to: {insights_root}")
    print("Generated tables:")
    for name in sorted(tables):
        print(f"- {name}.csv")


if __name__ == "__main__":
    main()
