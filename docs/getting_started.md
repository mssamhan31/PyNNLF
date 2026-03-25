# Installation Instructions

1. Install the package.

    On macOS, use `python3`/`pip3` if `python`/`pip` are not available.

    ```bash
    python -m pip install pynnlf
    ```

This may take about 10 minutes. Although newer Python versions may work, the tool was tested on Python 3.12.3.

# How to Use The Tool
1. Initialize a workspace in any directory you want. This guide uses `example_project` as the example workspace name. By default, only the sample dataset (`ds0`) is included.

    Run this in Python. On macOS, start Python with `python3` if `python` is not available.

    ```python
    import pynnlf

    pynnlf.init("example_project")
    ```

2. Set up your experiment in `example_project/specs/experiment.yaml`.

3. Run the experiment.

    ```python
    import pynnlf

    pynnlf.run_experiment("example_project/specs/experiment.yaml")
    ```

4. To skip plot generation and save storage for a run, pass `plot_enabled=False`.

    ```python
    import pynnlf

    pynnlf.run_experiment(
        "example_project/specs/experiment.yaml",
        plot_enabled=False,
    )
    ```

5. The tool outputs evaluation results in `example_project/experiment_result/`. When plotting is disabled, PyNNLF does not create the `*_cv1_plots/` folder.

Example YAML inputs in `example_project/specs/experiment.yaml`:

```yaml
dataset: ds0
forecast_horizon: fh1
model: m6
hyperparameter: hp1
```

For the list of available datasets and models, how to modify model hyperparameters, how to add a model, how to add a dataset, and the API reference, see the Detailed Guide page.
