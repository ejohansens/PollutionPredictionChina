from pathlib import Path

import pandas as pd

from air_quality.data import load_all_data
from air_quality.modeling import (
    summarize_cv_report,
    evaluate_time_holdout,
    plot_cv_and_holdout,
)
from air_quality.pipeline import MODELS


def main() -> None:

    workspace_root = Path(__file__).resolve().parent
    data_root = workspace_root / "PRSA_Data_20130301-20170228"
    output_root = workspace_root / "output"
    benchmark_root = output_root / "benchmark"

    print("Loading data...")
    df = load_all_data(data_root)

    for model_name in MODELS:

        print()
        print("=" * 60)
        print(f"Evaluating {model_name.upper()}")
        print("=" * 60)

        report_csv = benchmark_root / f"{model_name}_walk_forward.csv"

        if not report_csv.exists():
            print(f"Skipping {model_name}: file not found.")
            continue

        report_df = pd.read_csv(
            report_csv,
            parse_dates=[
                "train_start",
                "train_end",
                "val_start",
                "val_end",
            ],
        )

        model_output = benchmark_root / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        print("Summarizing cross-validation results...")
        summarize_cv_report(report_df, model_output, model_name=model_name)

        print("Running holdout evaluation...")
        holdout_df = evaluate_time_holdout(
            df=df,
            output_root=model_output,
            holdout_days=365,
            model_name=model_name,
        )

        print("Generating plots...")
        plot_cv_and_holdout(
            report_df,
            holdout_df,
            model_output,
        )

    print()
    print("Evaluation completed.")
    print(f"Artifacts saved to: {benchmark_root}")


if __name__ == "__main__":
    main()