# CI (Smoke Test)

GitHub Actions CI is included to run a smoke test of 3 models and check results against the standard benchmark. This runs automatically on push and pull request.

# Local Testing

## 1) Simple local test (single experiment)
1. Initialize a workspace.

    Run this in Python. On macOS, start Python with `python3` if `python` is not available.

    ```python
    import pynnlf

    pynnlf.init("example_project")
    ```

2. Edit `example_project/specs/experiment.yaml` with your inputs, for example:

    ```yaml
    dataset: ds0
    forecast_horizon: fh1  # fh1 = 30 minutes ahead
    model: m6              # m6 = Linear Regression
    hyperparameter: hp1
    ```

3. Run the experiment.

    ```python
    import pynnlf

    pynnlf.run_experiment("example_project/specs/experiment.yaml")
    ```

    To skip plot generation and save storage:

    ```python
    import pynnlf

    pynnlf.run_experiment(
        "example_project/specs/experiment.yaml",
        plot_enabled=False,
    )
    ```

4. Check outputs under `example_project/experiment_result/`.

## 2) Local smoke test (3 models)
This runs the same 3-model check used in CI and compares results to the benchmark.

1. Initialize a workspace if you have not already:

    ```python
    import pynnlf

    pynnlf.init("example_project")
    ```

2. Run the smoke test.

    ```python
    import pynnlf

    pynnlf.run_tests("example_project/specs/tests_ci.yaml")
    ```

    To skip plot generation during the test run:

    ```python
    import pynnlf

    pynnlf.run_tests(
        "example_project/specs/tests_ci.yaml",
        plot_enabled=False,
    )
    ```

3. Check the report in `example_project/experiment_result/Archive/Testing Result/`.

This also writes `example_project/experiment_result/a1_experiment_result.csv`. Recap skips missing or malformed CSVs with warnings. The test report compares only the experiments created by that test invocation.

## 3) Full model test (18 models)
This runs all 18 models and compares results to the benchmark.

1. Initialize a workspace if you have not already:

    ```python
    import pynnlf

    pynnlf.init("example_project")
    ```

2. Run the full test.

    ```python
    import pynnlf

    pynnlf.run_tests("example_project/specs/tests_full.yaml")
    ```

    To skip plot generation during the full test:

    ```python
    import pynnlf

    pynnlf.run_tests(
        "example_project/specs/tests_full.yaml",
        plot_enabled=False,
    )
    ```

3. Check the report in `example_project/experiment_result/Archive/Testing Result/`.

Estimated time: about 1 hour on a personal computer with an Intel i5 processor and 32GB RAM. Example historical results can be found in:

```text
experiment_result/Archive/Testing Result/20250821_test_result_CEEM Computer.csv
experiment_result/Archive/Testing Result/20250821_test_result_UNSW Laptop.csv
experiment_result/Archive/Testing Result/20250822_test_result_SS Personal Laptop
```

# Unit Tests

The repository also carries a small pytest suite covering the package's core guarantees: that it installs and creates a workspace, that an experiment runs end to end and writes a result table with the expected columns, and that the error metrics return known values.

```bash
python -m pip install -e ".[test]"
python -m pytest
```

These run in continuous integration on every push and pull request, alongside `ruff` linting and the smoke test described above.
