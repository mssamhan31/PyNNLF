# Load composition comparison, test RMSE in kilowatts

Mean test RMSE across 10 cross-validation folds, plus or minus the fold
standard deviation. Forecast horizon 1 day. Models ordered by RMSE on the
underlying load.

| Model | Underlying load (kW) | Net load with PV (kW) | Net load with PV and battery (kW) |
|---|---|---|---|
| lr_hp1 | 6.58 +/- 1.38 | 14.96 +/- 1.92 | 13.59 +/- 2.30 |
| xgb_hp1 | 6.75 +/- 1.46 | 16.07 +/- 2.56 | 14.65 +/- 2.67 |
| rf_hp1 | 7.18 +/- 1.42 | 17.63 +/- 2.46 | 15.55 +/- 2.39 |
| dnn_hp1 | 7.24 +/- 1.54 | 15.73 +/- 1.60 | 14.24 +/- 2.28 |
| ann_hp1 | 7.51 +/- 1.65 | 15.55 +/- 1.69 | 14.06 +/- 2.12 |
| ets_hp1 | 7.67 +/- 1.64 | 17.16 +/- 2.86 | 15.28 +/- 3.03 |
| naive_hp1 | 7.84 +/- 1.63 | 17.02 +/- 2.73 | 15.23 +/- 2.86 |
| snaive_hp2 | 9.20 +/- 2.11 | 21.66 +/- 2.69 | 19.88 +/- 3.27 |
| prophet_hp1 | 9.23 +/- 1.60 | 21.10 +/- 3.93 | 19.38 +/- 4.12 |
| rt_hp3 | 9.65 +/- 1.76 | 21.94 +/- 1.98 | 20.15 +/- 2.72 |
| arima_hp1 | 14.40 +/- 2.41 | 52.16 +/- 7.96 | 41.37 +/- 8.81 |
| lstm_hp2 | 15.05 +/- 2.76 | 63.41 +/- 8.75 | 47.92 +/- 9.40 |
