from pathlib import Path

from air_quality.analysis import (
    generate_main_report,
    save_analysis_outputs,
)

from air_quality.data import load_all_data

from air_quality.modeling import (
    evaluate_models_walk_forward,
    compare_models,
)


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

    print("Running forecasting benchmark (XGBoost, LightGBM, CatBoost, Ridge, ElasticNet, Hybrid, Stacking)...")

    results = evaluate_models_walk_forward(
        df=df,
        output_root=output_root / "benchmark",
        models=["xgboost", "lightgbm", "catboost", "ridge", "elasticnet", "hybrid", "stacking"],
    )

    print("Comparing models...")
    summary = compare_models(results, output_root / "benchmark")

    summary.to_csv(output_root / "benchmark" / "model_summary.csv", index=False)

    print("Analysis and forecasting complete.")
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()