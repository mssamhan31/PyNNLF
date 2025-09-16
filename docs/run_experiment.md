# How to Use The Tool
- Open the notebook: `notebooks/model/run_experiments.ipynb.`
- Fill in the input values for your forecast problem and model specification. Example:
```
dataset = "ds0"            # Dataset for testing
forecast_horizon = "fh1"   # fh1 = 30 minutes ahead
model_name = "m6"          # Linear regression
hyperparameter_no = "hp1"  # Hyperparameter ID
```
- Run the notebook.

- The tool outputs evaluation results to the `experiment_result/folder`.

- To evaluate multiple forecast problems and model specifications at once, use `notebooks/model/run_experiments_batch.ipynb`

For the list of available datasets & models, how to modify model hyperparameter, how to add a model, how to add a dataset, and exhaustive list of API Reference see the Detailed Guide page.