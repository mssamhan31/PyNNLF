# How to Use The Tool
1. Initialize a workspace. This page uses `example_project` as the example workspace name. By default, only the sample dataset (`ds0`) is included.

    Run this in Python. On macOS, start Python with `python3` if `python` is not available.

    ```python
    import pynnlf

    pynnlf.init("example_project")
    ```

2. Fill in the input values for your forecast problem and model specification in `example_project/specs/experiment.yaml`.

    ```yaml
    dataset: ds0
    forecast_horizon: fh1  # fh1 = 30 minutes ahead
    model: m6              # Linear regression
    hyperparameter: hp1    # Hyperparameter ID
    ```

3. Run the experiment.

    ```python
    import pynnlf

    pynnlf.run_experiment("example_project/specs/experiment.yaml")
    ```

4. To skip plot generation and save storage, use the per-run override.

    ```python
    import pynnlf

    pynnlf.run_experiment(
        "example_project/specs/experiment.yaml",
        plot_enabled=False,
    )
    ```

5. Run a batch of experiments (optional).

    ```python
    import pynnlf

    pynnlf.run_experiment_batch("example_project/specs/batch.yaml")
    ```

    To skip plot generation for every experiment in the batch:

    ```python
    import pynnlf

    pynnlf.run_experiment_batch(
        "example_project/specs/batch.yaml",
        plot_enabled=False,
    )
    ```

6. Recap multiple experiments into one CSV (optional).

    ```python
    import pynnlf

    pynnlf.recap_experiments("example_project/experiment_result")
    ```

    This writes `example_project/experiment_result/a1_experiment_result.csv` and skips missing or malformed CSVs with warnings.

7. The tool outputs evaluation results to `example_project/experiment_result/`. The `*_cv1_plots/` folder is created only when plot generation is enabled.

For the list of available datasets and models, how to modify model hyperparameters, how to add a model, how to add a dataset, and the API reference, see the Detailed Guide page.
