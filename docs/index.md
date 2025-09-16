# Welcome to PyNNLF
PyNNLF (Python for Network Net Load Forecast) is a tool to evaluate net load forecasting model performance in a reliable and reproducible way.

You can access the [GitHub repository here](https://github.com/mssamhan31/PyNNLF).

# Objective
This tool evaluates net load forecasting models reliably and reproducibly. It includes a library of public net load datasets and common forecasting models, including simple benchmark models. Users input the forecast problem and model specification, and the tool outputs evaluation results. 

It also allows users to add datasets, models, and modify hyperparameters. Researchers claiming a new or superior model can compare their model with existing ones on public datasets. The target audience includes researchers in academia or industry focused on evaluating and optimizing net load forecasting models. 

A visual illustration of the tool workflow is shown below.
![Home Illustration](img/home_illustration.png)

# Input
1. **Forecast Target**: dataset & forecast horizon. List of possible forecast problem values is in `notebooks/config/config.ipynb`.
2. **Model Specification**: model & hyperparameters. List of possible model specifications is in `notebooks/config/model_hyperparameters.ipynb`.

# Output
1. `a1_experiment_result.csv` – contains accuracy (cross-validated test n-RMSE), stability (accuracy stddev), and training time.
2. `a2_hyperparameter.csv` – lists hyperparameters used for each model.
3. `a3_cross_validation_result.csv` – detailed results for each cross-validation split.
4. `cv_plots/` – folder with plots including:
   - Observation vs forecast (time plot)
   - Observation vs forecast (scatter plot)
   - Residual time plot
   - Residual histogram
5. `cv_test/` and `cv_train/` – folders containing time series of observation, forecast, and residuals for each cross-validation split.

# Tool Output Naming Convention
Format:
`[experiment_no]_[experiment_date]_[dataset]_[forecast_horizon]_[model]_[hyperparameter]`

Example:
`E00001_250915_ds0_fh30_m6_lr_hp1`
