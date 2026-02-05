# CI (Smoke Test)

GitHub Actions CI is included to run a smoke test of 3 models and check results against the standard benchmark. This runs automatically on push and pull request.

# Local Testing

## 1) Simple local test (single experiment)
1. Initialize a workspace:
    On macOS, use `python3 -c` if `python -c` is not available.

    ```
    python -c "import pynnlf; pynnlf.init('my_project')"
    ```

2. Edit `specs/experiment.yaml` with your inputs, for example:
    ```
    dataset: ds0
    forecast_horizon: fh1   # fh1 = 30 minutes ahead
    model: m6               # m6 = Linear Regression
    hyperparameter: hp1
    ```

3. Run the experiment:
    ```
    python -c "import pynnlf; pynnlf.run_experiment('my_project/specs/experiment.yaml')"
    ```

4. Check outputs under `my_project/experiment_result/`.

## 2) Local smoke test (3 models)
This runs the same 3-model check used in CI and compares results to the benchmark.

1. Initialize a workspace (if you have not already):
    ```
    python -c "import pynnlf; pynnlf.init('my_project')"
    ```

2. Run the smoke test:
    ```
    python -c "import pynnlf; pynnlf.run_tests('my_project/specs/tests_ci.yaml')"
    ```

3. Check the report in `my_project/experiment_result/Archive/Testing Result/`.

This also writes `my_project/experiment_result/a1_experiment_result.csv`. Recap skips missing or malformed CSVs with warnings.

## 3) Full model test (18 models)
This runs all 18 models and compares results to the benchmark.

1. Initialize a workspace (if you have not already):
    ```
    python -c "import pynnlf; pynnlf.init('my_project')"
    ```

2. Run the full test:
    ```
    python -c "import pynnlf; pynnlf.run_tests('my_project/specs/tests_full.yaml')"
    ```

3. Check the report in `my_project/experiment_result/Archive/Testing Result/`.

⏱ **Estimated time**: ~1 hour on a personal computer with an Intel i5 processor and 32GB RAM. Example historical results can be found in:
```
experiment_result/Archive/Testing Result/20250821_test_result_CEEM Computer.csv
experiment_result/Archive/Testing Result/20250821_test_result_UNSW Laptop.csv
experiment_result/Archive/Testing Result/20250822_test_result_SS Personal Laptop
```
