from pathlib import Path

from air_quality.data import load_all_data
from air_quality.modeling import (
    summarize_cv_report,
    evaluate_time_holdout,
    plot_cv_and_holdout,
)
import pandas as pd


def main():
    workspace_root = Path(__file__).resolve().parent
    data_root = workspace_root / "PRSA_Data_20130301-20170228"
    output_root = workspace_root / "output"

    print("Loading data...")
    df = load_all_data(data_root)

    report_csv = output_root / "xgboost_walk_forward_results.csv"
    if report_csv.exists():
        print(f"Loading CV report from {report_csv}")
        report_df = pd.read_csv(report_csv, parse_dates=["train_start", "train_end", "val_start", "val_end"])  # type: ignore
    else:
        raise FileNotFoundError(f"CV report not found at {report_csv}; run main.py first to generate it.")

    print("Summarizing CV report...")
    summarize_cv_report(report_df, output_root)

    print("Running time-based holdout evaluation (default 365 days)...")
    holdout_df = evaluate_time_holdout(df, output_root, holdout_days=365)

    print("Plotting comparisons...")
    plot_cv_and_holdout(report_df, holdout_df, output_root)

    print("All evaluation artifacts saved to:", output_root)


if __name__ == "__main__":
    main()
