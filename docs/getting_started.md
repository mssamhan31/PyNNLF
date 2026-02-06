# Installation Instructions

1. Install the package:

    On macOS, use `python3`/`pip3` if `python`/`pip` are not available.

    ```
    python -m pip install pynnlf
    ```

⚠️ This may take ~10 minutes.
Although newer Python versions may work, the tool was tested on Python 3.12.3.


# How to Use The Tool
1. Initialize a workspace in any directory you want (example name: `my_project`). By default, only the sample dataset (ds0) is included. You can choose to download all datasets if needed:

    On macOS, use `python3 -c` if `python -c` is not available.

    ```
    python -c "import pynnlf; pynnlf.init('my_project')"
    ```

2. Set up your experiment in `specs/experiment.yaml`.

3. Run the experiment:

    ```
    python -c "import pynnlf; pynnlf.run_experiment('my_project/specs/experiment.yaml')"
    ```

4. The tool outputs evaluation results in the `experiment_result/` folder inside your workspace.

Example YAML inputs in `specs/experiment.yaml`:
```
dataset: ds0
forecast_horizon: fh1
model: m6
hyperparameter: hp1
```

For the list of available datasets & models, how to modify model hyperparameter, how to add a model, how to add a dataset, and exhaustive list of API Reference see the Detailed Guide page.