# How to Use The Tool
1. Initialize a workspace (example name: `my_project`). By default, only the sample dataset (ds0) is included. You can choose to download all datasets if needed:

    On macOS, use `python3 -c` if `python -c` is not available.

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
    python -c "import pynnlf; pynnlf.run_experiment('example_project/specs/experiment.yaml')"
    ```

4. Run a batch of experiments (optional):

    ```
    python -c "import pynnlf; pynnlf.run_experiment_batch('example_project/specs/batch.yaml')"
    ```

5. Recap multiple experiments into one CSV (optional):

    ```
    python -c "import pynnlf; pynnlf.recap_experiments('example_project/experiment_result')"
    ```

    This writes `example_project/experiment_result/a1_experiment_result.csv` and skips missing or malformed CSVs with warnings.

6. The tool outputs evaluation results to the `experiment_result/` folder inside your workspace.

For the list of available datasets & models, how to modify model hyperparameter, how to add a model, how to add a dataset, and exhaustive list of API Reference see the Detailed Guide page.