# Interesting insights from existing outputs

This report is generated only from files already saved in `output/`. It does not retrain models.

## Best model by forecast horizon

- 1h: catboost with MAE=9.905 and RMSE=18.563.
- 3h: xgboost with MAE=21.512 and RMSE=37.078.
- 6h: xgboost with MAE=33.958 and RMSE=53.830.
- 12h: lightgbm with MAE=49.320 and RMSE=72.805.
- 24h: xgboost with MAE=67.717 and RMSE=91.106.

## Forecast degradation from shortest to longest horizon

- lightgbm: MAE ratio longest/shortest = 6.72 (delta=57.930).
- xgboost: MAE ratio longest/shortest = 6.75 (delta=57.691).
- catboost: MAE ratio longest/shortest = 6.91 (delta=58.566).

## Features with strongest consensus across models

- PM2.5: average normalized importance 58.97% across 3 model(s).
- PM10: average normalized importance 5.13% across 3 model(s).
- PM2.5_lag1: average normalized importance 2.81% across 3 model(s).
- DEWP: average normalized importance 2.66% across 3 model(s).
- PM10_lag1: average normalized importance 2.38% across 3 model(s).
- WSPM: average normalized importance 1.97% across 3 model(s).
- DEWP_lag3: average normalized importance 1.59% across 3 model(s).
- No: average normalized importance 1.22% across 3 model(s).
- CO: average normalized importance 1.17% across 3 model(s).
- NO2: average normalized importance 1.10% across 3 model(s).

## Stations with highest average PM2.5

- Dongsi: mean PM2.5 = 86.19.
- Wanshouxigong: mean PM2.5 = 85.02.
- Nongzhanguan: mean PM2.5 = 84.84.
- Gucheng: mean PM2.5 = 83.85.
- Wanliu: mean PM2.5 = 83.37.

## Seasonal PM2.5 pattern

- Highest average PM2.5 month: 12, mean PM2.5 = 103.92.
