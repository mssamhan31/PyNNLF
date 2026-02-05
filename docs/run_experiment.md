# How to Use The Tool
1. Initialize a workspace (example name: `my_project`). By default, only the sample dataset (ds0) is included. You can choose to download all datasets if needed:
```
python -c "import pynnlf; pynnlf.init('my_project')"
```

2. Fill in the input values for your forecast problem and model specification in `specs/experiment.yaml`. Example:
```
dataset: ds0
forecast_horizon: fh1   # fh1 = 30 minutes ahead
model: m6               # Linear regression
hyperparameter: hp1     # Hyperparameter ID
```

3. Run the experiment:
```
python -c "import pynnlf; pynnlf.run_experiment('my_project/specs/experiment.yaml')"
```

4. The tool outputs evaluation results to the `experiment_result/` folder inside your workspace.

For the list of available datasets & models, how to modify model hyperparameter, how to add a model, how to add a dataset, and exhaustive list of API Reference see the Detailed Guide page.