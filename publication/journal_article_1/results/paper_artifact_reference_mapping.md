# Paper Artifact Reference Mapping

This file maps generated CSV/PNG artifacts to their intended manuscript section and proposed label.

## Methods - Experiment setup
- Table M1: results/paper_table_experiment_parameters.csv - Experiment parameters: predictors, lags, horizons, CV, models, metrics

## Methods - PyNNLF workflow outputs
- Table M2: results/paper_table_pynnlf_output_artifacts.csv - PyNNLF output files, contents, and purpose

## Results - Model selection recommendation
- Table M3: results/paper_table_model_selection_recommendation.csv - Pareto, utility, and satisficing recommendation outputs

## Results - Aggregation level comparison
- Figure A1: results/03_aedp_aggregation_level/figures/fig10_aedp_agg_nrmse_mean_std_cvbg.png - Aggregation nRMSE: sample mean +- SD with fold background
- Figure A2: results/03_aedp_aggregation_level/figures/fig11_aedp_agg_rmse_per_hh_mean_std_cvbg.png - Aggregation RMSE per household: sample mean +- SD with fold background
- Figure A3(a): results/03_aedp_aggregation_level/figures/fig12a_aedp_agg1_actual_vs_forecast_timeseries_naive_lr_xgb.png - Actual vs forecast (1 household)
- Figure A3(b): results/03_aedp_aggregation_level/figures/fig12b_aedp_agg10_actual_vs_forecast_timeseries_naive_lr_xgb.png - Actual vs forecast (10 households)
- Figure A3(c): results/03_aedp_aggregation_level/figures/fig12c_aedp_agg100_actual_vs_forecast_timeseries_naive_lr_xgb.png - Actual vs forecast (100 households)
- Figure A3(d): results/03_aedp_aggregation_level/figures/fig12d_aedp_agg1000_actual_vs_forecast_timeseries_naive_lr_xgb.png - Actual vs forecast (1000 households)
- Figure A4(a): results/03_aedp_aggregation_level/figures/fig13a_aedp_agg1_actual_vs_forecast_scatter_naive_lr_xgb.png - Scatter actual vs forecast (1 household)
- Figure A4(b): results/03_aedp_aggregation_level/figures/fig13b_aedp_agg10_actual_vs_forecast_scatter_naive_lr_xgb.png - Scatter actual vs forecast (10 households)
- Figure A4(c): results/03_aedp_aggregation_level/figures/fig13c_aedp_agg100_actual_vs_forecast_scatter_naive_lr_xgb.png - Scatter actual vs forecast (100 households)
- Figure A4(d): results/03_aedp_aggregation_level/figures/fig13d_aedp_agg1000_actual_vs_forecast_scatter_naive_lr_xgb.png - Scatter actual vs forecast (1000 households)

## Results - Load composition comparison
- Figure B1: results/04_sa_bess_clean_44hh/figures/fig20_sa_bess_composition_actual_vs_forecast_timeseries.png - Time series actual vs forecast across load compositions with reading-based PV and battery overlays
- Figure B2: results/04_sa_bess_clean_44hh/figures/fig21_sa_bess_composition_error_timeseries.png - Forecast error time series across load compositions

## Results - Dataset comparison
- Figure C1: results/01_ashd_aedp_148hh_comparison/figures/fig30_ashd_vs_aedp_xgb_actual_vs_forecast_daypair.png - XGBoost actual vs forecast for ASHD vs AEDP (representative 1-week windows) with peak-demand lines
- Figure C2: results/01_ashd_aedp_148hh_comparison/figures/fig31_ashd_vs_aedp_xgb_error_profile_daypair.png - XGBoost error profile for ASHD vs AEDP

## Results - Forecast horizon comparison
- Figure D1: results/02_ashd_148hh_forecast_horizon/figures/fig40_ashd_horizons_xgb_actual_vs_forecast.png - XGBoost actual vs forecast for 30-min, 1-day, and 1-week horizons
- Figure D2: results/02_ashd_148hh_forecast_horizon/figures/fig41_ashd_horizons_xgb_error_summary.png - XGBoost horizon error summary

