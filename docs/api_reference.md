# API Reference

This page is split into two sections:
1) **User-facing API** — functions you call directly.
2) **Internal API** — background modules used by the engine.

## 1) User-facing API

### `init(workspace_dir, *, download_data=False, all_data=False, datasets=None, base_url_override=None)`
Create a workspace by copying the scaffold and optionally downloading datasets.

Args:

- `workspace_dir`: target directory to create the workspace.

- `download_data`: whether to download datasets from the online source.

- `all_data`: when `True`, download all datasets except `ds0`.

- `datasets`: explicit list of dataset IDs to download (e.g., `['ds15']`).

- `base_url_override`: override the download base URL.

Returns:

- `Path`: workspace directory path.

### `run_experiment(spec_path, *, plot_enabled=None)`
Run a single experiment from a workspace `specs/experiment.yaml`.

Args:

- `spec_path`: path to the YAML spec file.

- `plot_enabled`: optional per-run override for `pynnlf_config.yaml`. When `False`, PyNNLF skips plot generation and does not create the `*_cv1_plots/` folder.

Returns:

- `None`.

### `run_experiment_batch(spec_path, *, plot_enabled=None)`
Run a batch of experiments from a workspace `specs/batch.yaml`.

Args:

- `spec_path`: path to the YAML batch spec file.

- `plot_enabled`: optional per-run override for `pynnlf_config.yaml`. When `False`, PyNNLF skips plot generation and does not create the `*_cv1_plots/` folder.

Returns:

- `None`.

### `recap_experiments(results_root='experiment_result', output_path=None, return_df=False, include_experiment_folder=True)`
Aggregate experiment summary CSVs into a single recap file.

Args:

- `results_root`: root folder containing `E00001_*` experiment directories.

- `output_path`: output CSV path (default: `<results_root>/a1_experiment_result.csv`).

- `return_df`: when `True`, return a `pd.DataFrame` instead of a path.

- `include_experiment_folder`: add `experiment_folder` column to output.

Returns:

- `Path` or `pd.DataFrame` depending on `return_df`.

### `run_tests(spec_path, *, plot_enabled=None)`
Run regression tests defined in `specs/tests_ci.yaml` or `specs/tests_full.yaml` and write a report.

Args:

- `spec_path`: path to the YAML tests spec file.

- `plot_enabled`: optional per-run override for `pynnlf_config.yaml`. When `False`, the test run skips plot generation and does not create the `*_cv1_plots/` folder.

Returns:

- `Path`: path to the generated report CSV.

## 2) Internal API (advanced)

### Module: `runner`

#### `run_single(spec_path, *, plot_enabled=None)`
Run one experiment from a YAML spec.

Args:

- `spec_path`: path to `specs/experiment.yaml`.

- `plot_enabled`: optional per-run override for `pynnlf_config.yaml`.

Returns:

- `None`.

#### `run_batch(spec_path, *, plot_enabled=None)`
Run batch experiments from a YAML batch spec (cartesian product).

Args:

- `spec_path`: path to `specs/batch.yaml`.

- `plot_enabled`: optional per-run override for `pynnlf_config.yaml`.

Returns:

- `None`.

#### `_workspace_root_from_spec(spec_path)`
Infer the workspace root from a spec path.

Args:

- `spec_path`: path to a spec file under `specs/`.

Returns:

- `Path`: workspace root directory.

### Module: `tests_runner`

#### `run_tests(spec_path, *, plot_enabled=None)`
Run regression tests and compare results to the benchmark.

Args:

- `spec_path`: path to `specs/tests_*.yaml`.

- `plot_enabled`: optional per-run override for `pynnlf_config.yaml`.

Returns:

- `Path`: report CSV path.

#### `_workspace_root_from_spec(spec_path)`
Infer the workspace root from a tests spec path.

Args:

- `spec_path`: path to a tests spec file.

Returns:

- `Path`: workspace root directory.

#### `_find_latest_experiment_dir(output_dir)`
Find the most recent experiment folder matching `E00001_*`.

Args:

- `output_dir`: workspace `experiment_result/` directory.

Returns:

- `Path`: latest experiment directory.

#### `_load_benchmark_csv(workspace_root, benchmark_rel)`
Load the benchmark CSV and index it by `metric_id`.

Args:

- `workspace_root`: workspace root directory.

- `benchmark_rel`: relative benchmark CSV path (e.g., `specs/testing_benchmark.csv`).

Returns:

- `pd.DataFrame`: benchmark table indexed by `metric_id`.

#### `_write_report(report_path, report_rows)`
Write a report CSV in the standard test format.

Args:

- `report_path`: output file path.

- `report_rows`: list of report rows (dicts).

Returns:

- `None`.

### Module: `discovery`

#### `discover_unique_file(directory, prefix, suffix)`
Discover a unique file by prefix and suffix (e.g., `m19_*.py`).

Args:

- `directory`: directory to search.

- `prefix`: required filename prefix.

- `suffix`: required filename suffix.

Returns:

- `Path`: matching file path.

#### `discover_model_name(models_dir, model_id)`
Find a model file and return its stem (e.g., `m19_my_model`).

Args:

- `models_dir`: workspace `models/` directory.

- `model_id`: model ID from spec (e.g., `m6`).

Returns:

- `str`: model file stem.

#### `discover_dataset_path(data_dir, dataset_id)`
Find a dataset CSV path by dataset ID.

Args:

- `data_dir`: workspace `data/` directory.

- `dataset_id`: dataset ID from spec (e.g., `ds0`).

Returns:

- `Path`: dataset file path.

### Module: `hyperparams`

#### `load_hyperparameters(path)`
Load model hyperparameters from YAML.

Args:

- `path`: path to `models/hyperparameters.yaml`.

Returns:

- `dict`: `{model_name: {hp_no: hyperparameter_dict}}`.

#### `get_hp(hparams, model_name, hp_no)`
Retrieve a hyperparameter dict for a model and `hp_no`.

Args:

- `hparams`: output of `load_hyperparameters()`.

- `model_name`: model file stem (e.g., `m6_lr`).

- `hp_no`: hyperparameter ID (e.g., `hp1`).

Returns:

- `dict`: hyperparameter dict.

### Module: `yamlio`

#### `load_yaml(path)`
Load a YAML file into a dictionary using `safe_load`.

Args:

- `path`: path to a `.yaml`/`.yml` file.

Returns:

- `dict`: parsed YAML content.

### Module: `model_utils` (legacy JSON workflow)

#### `_load_json(path)`
Load a JSON file into a dictionary.

Args:

- `path`: path to a JSON file.

Returns:

- `dict`: parsed JSON content.

#### `_workspace_root_from_spec(spec_path)`
Infer workspace root from a JSON spec path.

Args:

- `spec_path`: path to `specs/experiment.json` or `specs/batch.json`.

Returns:

- `Path`: workspace root directory.

#### `run_single(spec_path)`
Run one experiment from a legacy JSON spec.

Args:

- `spec_path`: path to `specs/experiment.json`.

Returns:

- `None`.

#### `run_batch(spec_path)`
Run batch experiments from a legacy JSON spec.

Args:

- `spec_path`: path to `specs/batch.json`.

Returns:

- `None`.

#### `remove_jump_df(train_df_y)`
Remove a time gap jump from a time series.

Args:

- `train_df_y`: time series data.

Returns:

- `pd.Series`: time series with the jump removed.

#### `separate_lag_and_exogenous_features(train_df_X, target_column='y', lag_prefix='y_lag')`
Split lag features and exogenous variables.

Args:

- `train_df_X`: dataframe with lag and exogenous features.

- `target_column`: target column name.

- `lag_prefix`: lag feature prefix.

Returns:

- `tuple`: `(X_lags, X_exog)` dataframes.

### Module: `engine`

#### `load_model_module(models_dir, model_name)`
Load a model module from the workspace `models/` directory.

Args:

- `models_dir`: workspace `models/` directory.

- `model_name`: model file stem (e.g., `m6_lr`).

Returns:

- `module`: imported model module.

#### `compute_exp_no(path_result)`
Compute the next experiment number and formatted ID.

Args:

- `path_result`: `experiment_result/` directory.

Returns:

- `tuple`: `(experiment_no, experiment_no_str)`.

#### `compute_folder_name(experiment_no_str, dataset_file, forecast_horizon, model_name, hyperparameter_no)`
Generate the experiment folder name using the standard convention.

Args:

- `experiment_no_str`: experiment ID string (e.g., `E00001`).

- `dataset_file`: dataset filename.

- `forecast_horizon`: forecast horizon in minutes.

- `model_name`: model file stem.

- `hyperparameter_no`: hyperparameter ID.

Returns:

- `str`: folder name.

#### `prepare_directory(path_result, dataset_file, forecast_horizon, model_name, hyperparameter_no, hyperparameter_dict, *, plot_enabled)`
Create experiment folders and file paths for outputs.

Args:

- `path_result`: `experiment_result/` directory.

- `dataset_file`: dataset filename.

- `forecast_horizon`: forecast horizon in minutes.

- `model_name`: model file stem.

- `hyperparameter_no`: hyperparameter ID.

- `hyperparameter_dict`: hyperparameter dict.

- `plot_enabled`: whether plot file paths and the `*_cv1_plots/` folder should be created.

Returns:

- `tuple`: `(hyperparameter_dict, experiment_no_str, filepath_dict)`.

#### `export_result(filepath, df_a1_result, cross_val_result_df, hyperparameter)`
Export summary CSVs for a1/a2/a3 outputs.

Args:

- `filepath`: dictionary of output paths.

- `df_a1_result`: a1 summary dataframe.

- `cross_val_result_df`: a3 cross-validation dataframe.

- `hyperparameter`: effective hyperparameter dict used for the run.

Returns:

- `None`.

#### `add_lag_features(df, forecast_horizon, max_lag_day)`
Add lagged features to the dataframe based on horizon and max lag days.

Args:

- `df`: input dataframe with datetime index and `y` column.

- `forecast_horizon`: horizon in minutes.

- `max_lag_day`: max lag depth in days.

Returns:

- `pd.DataFrame`: dataframe with lag features.

#### `separate_holdout(df, n_block)`
Split the dataset into CV data and a holdout block.

Args:

- `df`: cleaned dataframe with features and target.

- `n_block`: number of blocks (k + 1).

Returns:

- `tuple`: `(block_length, holdout_df, df)`.

#### `input_and_process(dataset_path, model_name, forecast_horizon, max_lag_day, n_block, hyperparameter)`
Load data, add lags and calendar features, and split into blocks.

Args:

- `dataset_path`: dataset CSV path.

- `model_name`: model file stem.

- `forecast_horizon`: horizon in minutes.

- `max_lag_day`: max lag depth in days.

- `n_block`: number of blocks.

- `hyperparameter`: hyperparameter dict.

Returns:

- `tuple`: `(block_length, holdout_df, df)`.

#### `split_time_series(df, cv_no, test_pct)`
Split into train and test sets for blocked CV.

Args:

- `df`: dataframe for CV.

- `cv_no`: CV fold index.

- `test_pct`: test fraction per fold.

Returns:

- `tuple`: `(train_df, test_df)`.

#### `split_xy(df)`
Split dataframe into predictors and target.

Args:

- `df`: dataframe containing `y` and predictors.

Returns:

- `tuple`: `(df_X, df_y)`.

#### `remove_jump_df(train_df_y)`
Remove time gaps from a time series.

Args:

- `train_df_y`: time series data.

Returns:

- `pd.Series`: cleaned time series.

#### `call_train(train_fn, hyperparameter, train_df_X, train_df_y, forecast_horizon)`
Call a model training function with a compatible signature.

Args:

- `train_fn`: model training function.

- `hyperparameter`: hyperparameter dict.

- `train_df_X`: training predictors.

- `train_df_y`: training target.

- `forecast_horizon`: horizon in minutes.

Returns:

- `object`: trained model object.

#### `call_forecast(forecast_fn, model, train_df_X, test_df_X, train_df_y, forecast_horizon)`
Call a model forecasting function with a compatible signature.

Args:

- `forecast_fn`: model forecast function.

- `model`: trained model object.

- `train_df_X`: training predictors.

- `test_df_X`: testing predictors.

- `train_df_y`: training target (optional for some models).

- `forecast_horizon`: horizon in minutes.

Returns:

- `tuple`: `(train_df_y_hat, test_df_y_hat)`.

#### `save_model(filepath, cv_no, model)`
Serialize and save a model to disk.

Args:

- `filepath`: output path dictionary.

- `cv_no`: CV fold index.

- `model`: trained model object.

Returns:

- `None`.

#### `to_series(y_hat, target_index)`
Align model output to a target index as a 1D series.

Args:

- `y_hat`: model output (array/Series/DataFrame).

- `target_index`: desired index.

Returns:

- `pd.Series`: aligned series.

#### `run_model(...)`
Run cross-validation, training, forecasting, and export outputs for one model.

Args:

- `df`: processed dataframe for CV.

- `model_mod`: loaded model module.

- `model_name`: model file stem.

- `hyperparameter`: hyperparameter dict.

- `filepath`: export paths.

- `forecast_horizon`: horizon in minutes.

- `experiment_no_str`: experiment ID string.

- `block_length`: weeks per block.

- `dataset_file`: dataset filename.

- `hyperparameter_no`: hyperparameter ID.

- `k`: number of CV folds.

- `test_pct`: test fraction.

- `train_pct`: train fraction.

- `n_block`: number of blocks.

- `plot_enabled`: whether to write plot PNGs for the first CV fold.

- `plot_style`: plot settings.

- `run_seed`: optional central run seed recorded in `a1`.

- `seed_keys_overridden`: optional seed-like hyperparameter keys overridden to `run_seed`.

Returns:

- `None`.

#### `validate_model_module(model_mod, model_name)`
Validate that a model module exposes required functions.

Args:

- `model_mod`: loaded model module.

- `model_name`: model file stem.

Returns:

- `None`.

#### `run_experiment_engine(dataset_path, forecast_horizon_min, model_name, hyperparameter_no, hyperparameter, output_dir, models_dir, config)`
Run a single experiment end-to-end using explicit inputs.

Args:

- `dataset_path`: dataset CSV path.

- `forecast_horizon_min`: forecast horizon in minutes.

- `model_name`: model file stem.

- `hyperparameter_no`: hyperparameter ID.

- `hyperparameter`: hyperparameter dict.

- `output_dir`: workspace `experiment_result/`.

- `models_dir`: workspace `models/`.

- `config`: parsed `pynnlf_config.yaml`.

Returns:

- `None`.

#### `compute_MBE(forecast, observation)`
Compute Mean Bias Error.

Args:

- `forecast`: forecast series.

- `observation`: observation series.

Returns:

- `float`: MBE.

#### `compute_MAE(forecast, observation)`
Compute Mean Absolute Error.

Args:

- `forecast`: forecast series.

- `observation`: observation series.

Returns:

- `float`: MAE.

#### `compute_RMSE(forecast, observation)`
Compute Root Mean Square Error.

Args:

- `forecast`: forecast series.

- `observation`: observation series.

Returns:

- `float`: RMSE.

#### `compute_MAPE(forecast, observation)`
Compute Mean Absolute Percentage Error.

Args:

- `forecast`: forecast series.

- `observation`: observation series.

Returns:

- `float`: MAPE.

#### `compute_MASE(forecast, observation, train_result)`
Compute Mean Absolute Scaled Error.

Args:

- `forecast`: forecast series.

- `observation`: observation series.

- `train_result`: training result dataframe containing `naive` and `observation`.

Returns:

- `float`: MASE.

#### `compute_fskill(forecast, observation, naive)`
Compute forecast skill against a naive baseline.

Args:

- `forecast`: forecast series.

- `observation`: observation series.

- `naive`: naive forecast series.

Returns:

- `float`: forecast skill.

#### `compute_R2(forecast, observation)`
Compute $R^2$ for scatter plot diagnostics.

Args:

- `forecast`: forecast series.

- `observation`: observation series.

Returns:

- `float`: $R^2$.

#### `timeplot_forecast(observation, forecast, pathname, dark_blue, orange)`
Save a time plot of observation vs forecast.

Args:

- `observation`: observation series.

- `forecast`: forecast series.

- `pathname`: output image path.

- `dark_blue`: color value.

- `orange`: color value.

Returns:

- `None`.

#### `scatterplot_forecast(observation, forecast, R2, pathname, dark_blue, orange)`
Save a scatter plot of observation vs forecast.

Args:

- `observation`: observation series.

- `forecast`: forecast series.

- `R2`: $R^2$ value.

- `pathname`: output image path.

- `dark_blue`: color value.

- `orange`: color value.

Returns:

- `None`.

#### `timeplot_residual(residual, pathname, dark_blue, orange)`
Save a residual time plot.

Args:

- `residual`: residual series.

- `pathname`: output image path.

- `dark_blue`: color value.

- `orange`: color value.

Returns:

- `None`.

#### `histogram_residual(residual, df, pathname, dark_blue, orange)`
Save a residual histogram.

Args:

- `residual`: residual series.

- `df`: full dataframe used to compute range.

- `pathname`: output image path.

- `dark_blue`: color value.

- `orange`: color value.

Returns:

- `None`.
