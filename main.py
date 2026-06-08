from pathlib import Path

from air_quality.analysis import (
    generate_main_report,
    save_analysis_outputs,
)
from air_quality.data import load_all_data
from air_quality.modeling import evaluate_xgboost_walk_forward


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    data_root = workspace_root / "PRSA_Data_20130301-20170228"
    output_root = workspace_root / "output"
    output_root.mkdir(exist_ok=True)

    print("Loading air quality data...")
    df = load_all_data(data_root)

    print(f"Loaded {len(df)} rows from {df['station'].nunique()} stations.")
    print("Generating exploratory summaries...")

    analysis = generate_main_report(df, output_root)
    save_analysis_outputs(df, analysis, output_root)

    print("Running XGBoost forecasting workflow...")
    evaluate_xgboost_walk_forward(df, output_root)

    print("Analysis and forecasting complete.")
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()
