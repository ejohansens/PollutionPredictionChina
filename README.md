# Air Quality Data Analysis Project

This project loads the PRSA air quality dataset from multiple station CSV files and performs a clean exploratory analysis in Python.

## What is included

- `main.py`: entry point for loading data, cleaning, summarizing, and generating reports.
- `air_quality/data.py`: data loading and cleaning utilities.
- `air_quality/pipeline.py`: forecasting preprocessing, lagged features, and imputation.
- `air_quality/modeling.py`: XGBoost training with time-aware walk-forward evaluation.
- `air_quality/analysis.py`: summary and station-level analysis functions.
- `requirements.txt`: Python dependencies.

## How to run

1. Open a terminal in `c:\Users\johan\AIBProject`
2. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the analysis and XGBoost forecasting workflow:

```powershell
python main.py
```

## Output

The script creates an `output/` folder with:

- `summary.csv`
- `station_summary.csv`
- `missing_report.csv`
- `monthly_summary.csv`
- `xgboost_walk_forward_results.csv`
- `xgboost_feature_importance.csv`
- plot images for overall PM2.5 and top stations
- `report.txt`
- `xgboost_report.txt`

## Notes

- This project is intentionally built without a machine-learning pipeline or feature engineering.
- It focuses on loading the data, cleaning it, and producing useful exploratory summaries.
