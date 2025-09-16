## Dataset Format

All datasets are stored in the `data/` folder in `.csv` format. Each file is named using the pattern `[dataset_id]_[dataset_name].csv`, e.g., `ds4_ashd_with_weather.csv`.

Some datasets may share the same net load data but differ in the availability of exogenous variables. For instance, `ds1_ashd.csv` is equivalent to `ds4_ashd_with_weather.csv` but without weather data.

Each CSV file must include two required columns: `datetime` and `netload_kW`.  
**PyNNLF uses `netload_kW` as the target variable for forecasting**, and automatically generates lag features based on it.

Any additional columns are treated as exogenous variables. These are also processed into lag features based on the forecast horizon, but are not used as targets.

Calendar features are excluded from the CSV files, as PyNNLF generates them dynamically during each experiment.

## How to Add a Dataset

To add a new dataset, simply create a `.csv` file in the `data/` folder following the naming convention above.  
Make sure to update `data/metadata.xlsx` to document the new dataset.
