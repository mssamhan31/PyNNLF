## Dataset Format

All datasets are stored in the workspace `data/` folder in `.csv` format. Each file is named using the pattern `[dataset_id]_[dataset_name].csv`, e.g., `ds4_ashd_with_weather.csv`.

Some datasets may share the same net load data but differ in the availability of exogenous variables. For instance, `ds1_ashd.csv` is equivalent to `ds4_ashd_with_weather.csv` but without weather data.

Each CSV file must include two required columns: `datetime` and `netload_kW`.  
**PyNNLF uses `netload_kW` as the target variable for forecasting**, and automatically generates lag features based on it.

Any additional columns are treated as exogenous variables. These are also processed into lag features based on the forecast horizon, but are not used as targets.

Calendar features are excluded from the CSV files, as PyNNLF generates them dynamically during each experiment.

## How to Add a Dataset

1. Create a `.csv` file in your workspace `data/` folder following the naming convention above.
2. Update `data/metadata.xlsx` to document the new dataset.
3. Use the dataset ID in your experiment spec:

```
dataset: ds19
```
