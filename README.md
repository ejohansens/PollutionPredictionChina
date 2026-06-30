# Short-Term PM2.5 Forecasting (Beijing PRSA)

This repository contains a full forecasting benchmark for PM2.5 using the PRSA Beijing multi-station dataset.

The workflow includes:
- data loading and cleaning,
- horizon-aware feature engineering,
- two-stage feature selection,
- walk-forward cross-validation,
- model comparison across 1h, 3h, 6h, 12h, and 24h horizons,
- feature importance analysis per model and per horizon.

## Project structure

- `main.py`: end-to-end entry point (analysis + benchmark + summary outputs)
- `air_quality/data.py`: CSV discovery, cleaning, parsing, and merge logic
- `air_quality/pipeline.py`: feature engineering, filtering, preprocessing transforms
- `air_quality/modeling.py`: model builders, walk-forward CV, importance generation, comparison plots
- `air_quality/analysis.py`: exploratory summaries and visual diagnostics

## Models benchmarked

- `ridge`
- `elasticnet`
- `xgboost`
- `lightgbm`
- `catboost`
- `hybrid` (Ridge + LightGBM residual learner)
- `stacking` (XGBoost + LightGBM + CatBoost with Ridge meta-learner)

## Reproducibility setup

1. Open terminal in `c:\Users\johan\AIBProject`
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Ensure dataset folder exists:

`PRSA_Data_20130301-20170228/`

5. Run full pipeline:

```powershell
python main.py
```



## Experimental coverage (rubric alignment)

Different tests were done in both stages.

### DPT / feature engineering tests
- Horizon-specific lag and rolling-window configurations
- Wind direction circular encoding (`wd_sin`, `wd_cos`)
- Domain features (`dew_depression`, `pm_fine_ratio`, `pm10_excess`)
- Rate-of-change features (`PM2.5_diff{lag}`)
- Rolling volatility features (`rollstd`)
- Correlation filter (`|r| >= 0.05`)
- Collinearity filter (`|r| <= 0.92` retained)
- Imputation/scaling strategy changes (`SimpleImputer` + `RobustScaler`)

### Model implementation tests
- Linear baselines: Ridge, ElasticNet
- Tree ensembles: XGBoost, LightGBM, CatBoost
- Hybrid model: linear + residual tree learner
- Stacking ensemble with meta-learner
- Time-aware walk-forward CV for each horizon
- Per-horizon feature importance extraction

## Main outputs

Generated runtime artifacts are written to `output/` (ignored by git):
- `output/benchmark/model_summary.csv`
- `output/benchmark/*_walk_forward.csv`
- `output/benchmark/*_importance.csv`
- `output/benchmark/model_comparison_mae.png`

## Notes

- All model evaluations use temporal splits (walk-forward) to avoid leakage.
- Random seeds are set in model builders where applicable for reproducibility.
