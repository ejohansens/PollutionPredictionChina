from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

sns.set(style="whitegrid")

# Summarize numeric measurements and count missing values.
def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df[NUMERIC_COLUMNS].copy()
    summary = numeric_df.describe().T
    summary["missing_count"] = numeric_df.isna().sum()
    summary["missing_percent"] = (summary["missing_count"] / len(df)) * 100
    return summary

# Compute mean values and row counts for each station.
def station_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("station")[NUMERIC_COLUMNS]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in NUMERIC_COLUMNS})
    )
    summary["rows"] = df.groupby("station").size()
    summary["missing_rows"] = df.groupby("station")[NUMERIC_COLUMNS].apply(lambda x: x.isna().any(axis=1).sum())
    return summary.sort_values(by="rows", ascending=False)

# Generate a missing-value count report for numeric columns.
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = (
        df[NUMERIC_COLUMNS]
        .isna()
        .sum()
        .rename("missing_count")
        .to_frame()
    )
    report["missing_percent"] = (report["missing_count"] / len(df)) * 100
    return report

# Aggregate numeric values by month for trend analysis.
def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("year_month")[NUMERIC_COLUMNS]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in NUMERIC_COLUMNS})
        .reset_index()
    )
    monthly["year_month"] = monthly["year_month"].astype(str)
    return monthly

# Create a line plot for average monthly PM2.5.
def plot_monthly_pm25(df: pd.DataFrame, output_path: Path) -> None:
    monthly = monthly_summary(df)
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x="year_month", y="mean_PM2.5")
    plt.xticks(rotation=45)
    plt.title("Average Monthly PM2.5")
    plt.xlabel("Year-Month")
    plt.ylabel("Mean PM2.5")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# Plot monthly PM2.5 trends for the top stations by mean pollution.
def plot_top_stations_pm25(df: pd.DataFrame, output_path: Path, top_n: int = 4) -> None:
    station_mean = df.groupby("station")["PM2.5"].mean().nlargest(top_n)
    top_stations = station_mean.index.tolist()
    subset = df[df["station"].isin(top_stations)].copy()
    subset["year_month"] = subset["date"].dt.to_period("M").astype(str)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=subset, x="year_month", y="PM2.5", hue="station", estimator="mean")
    plt.xticks(rotation=45)
    plt.title(f"Average Monthly PM2.5 for Top {top_n} Stations")
    plt.xlabel("Year-Month")
    plt.ylabel("Mean PM2.5")
    plt.legend(title="Station")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# Build the main analysis dictionary with summaries and reports.
def generate_main_report(df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "summary": summarize_numeric(df),
        "station_summary": station_summary(df),
        "missing_report": missing_report(df),
        "monthly_summary": monthly_summary(df),
    }

# Save analysis outputs and generate plots and a short text report.
def save_analysis_outputs(df: pd.DataFrame, analysis: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    analysis["summary"].to_csv(output_dir / "summary.csv")
    analysis["station_summary"].to_csv(output_dir / "station_summary.csv")
    analysis["missing_report"].to_csv(output_dir / "missing_report.csv")
    analysis["monthly_summary"].to_csv(output_dir / "monthly_summary.csv")

    plot_monthly_pm25(df, output_dir / "monthly_pm25.png")
    plot_top_stations_pm25(df, output_dir / "top_stations_pm25.png")

    report_lines = [
        f"Total rows: {len(df)}",
        f"Stations analyzed: {analysis['station_summary'].shape[0]}",
        f"Dataset start: {df['date'].min().date() if 'date' in df.columns else 'n/a'}",
        f"Dataset end: {df['date'].max().date() if 'date' in df.columns else 'n/a'}",
        "",
        "Review output files for details and visual summaries.",
    ]
    (output_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
