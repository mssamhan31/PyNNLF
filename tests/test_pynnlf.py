"""Minimum test suite for PyNNLF (Python for Network Net Load Forecasting).

Four tests, chosen for the failures that actually happen rather than for coverage:

1. The package imports and builds a workspace, which catches a broken install or a
   scaffold that was not packaged.
2. An experiment runs end to end on the bundled sample dataset and writes a result
   table with the expected columns, which catches a broken pipeline.
3. Two error metrics return hand-computed values, which catches a silent change to
   how accuracy is measured.
4. The two sequence models (N-HiTS, PatchTST) return a finite forecast of the right
   length and reproduce it under the same seed, which catches a broken lag-window
   reshape or a lost seed.

Inputs:  the bundled scaffold and its ds0 sample dataset; no external data or network.
Outputs: assertions only; every file is written under pytest's tmp_path.
"""

import pandas as pd
import pytest

import pynnlf


# --- 1. The package imports and creates a workspace -------------------------------

def test_init_creates_workspace(tmp_path):
    """pynnlf.init() must produce a workspace a user can immediately run from."""
    workspace = pynnlf.init(tmp_path / "ws")

    for expected in ("models", "data", "specs"):
        assert (workspace / expected).is_dir(), f"missing {expected}/ in the workspace"

    # The sample dataset and the experiment specification are what the quick start uses.
    assert (workspace / "data" / "ds0_test.csv").is_file()
    assert (workspace / "specs" / "experiment.yaml").is_file()
    assert (workspace / "models" / "hyperparameters.yaml").is_file()


# --- 2. The pipeline runs end to end ----------------------------------------------

# The a1 summary is what every downstream recap and benchmark comparison reads, so
# these column names are effectively a public interface.
EXPECTED_A1_COLUMNS = {
    "experiment_no",
    "dataset_no",
    "forecast_horizon_min",
    "model_name",
    "run_seed",
    "runtime_ms",
    "test_RMSE",
    "test_nRMSE",
}


def test_experiment_runs_and_writes_expected_columns(tmp_path):
    """A default experiment must produce an a1 summary with the expected columns."""
    workspace = pynnlf.init(tmp_path / "ws")

    pynnlf.run_experiment(workspace / "specs" / "experiment.yaml", plot_enabled=False)

    results = sorted((workspace / "experiment_result").glob("E*/*_a1_experiment_result.csv"))
    assert results, "no a1 experiment result was written"

    df = pd.read_csv(results[0])
    assert len(df) == 1, "the a1 summary should hold exactly one row per experiment"
    missing = EXPECTED_A1_COLUMNS - set(df.columns)
    assert not missing, f"a1 summary is missing columns: {sorted(missing)}"
    assert df["test_RMSE"].notna().all(), "test_RMSE was not computed"


# --- 3. Metrics return known values -----------------------------------------------

def test_metrics_match_hand_computed_values():
    """Pin two error metrics to values worked out by hand.

    forecast - observation is [-1, -2, -3], so:
      mean absolute error  = (1 + 2 + 3) / 3            = 2.0
      root mean square error = sqrt((1 + 4 + 9) / 3)    = 2.16 to 2 decimal places
    """
    from pynnlf.engine import compute_MAE, compute_RMSE

    forecast = pd.Series([1.0, 2.0, 3.0])
    observation = pd.Series([2.0, 4.0, 6.0])

    assert compute_MAE(forecast, observation) == pytest.approx(2.0)
    assert compute_RMSE(forecast, observation) == pytest.approx(2.16, abs=1e-3)


# --- 4. Sequence models honour the model contract -----------------------------------

@pytest.mark.parametrize("stem", ["m19_nhits", "m20_patchtst"])
def test_sequence_models_forecast_shape_and_determinism(stem):
    """N-HiTS and PatchTST must return finite 1-D forecasts and be seed-reproducible.

    Uses a small synthetic frame laid out like the engine's: weather columns, a
    newest-first ``y_lag_*`` block, then calendar dummies, one of which is constant.
    """
    import importlib
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(pynnlf.__file__).parent / "scaffold" / "models"))
    module = importlib.import_module(stem)
    train = getattr(module, f"train_model_{stem}")
    forecast = getattr(module, f"produce_forecast_{stem}")

    rng = np.random.default_rng(0)
    n_rows, lookback = 200, 96
    X = pd.DataFrame(rng.normal(size=(n_rows, lookback)), columns=[f"y_lag_{i}" for i in range(1, lookback + 1)])
    X.insert(0, "air_temperature_in_degrees_c", rng.normal(size=n_rows))
    X["weekday_1"] = rng.integers(0, 2, n_rows)
    X["constant_dummy"] = 1.0
    y = pd.DataFrame({"y": rng.normal(size=n_rows)})
    hyperparameter = {
        "seed": 1, "hidden_size": 16, "num_layers": 1, "pool_kernel_sizes": [4, 2, 1],
        "exog_hidden_size": 4, "patch_length": 16, "stride": 8, "d_model": 8, "num_heads": 2,
        "feedforward_size": 16, "dropout": 0.1, "batch_size": 64, "epochs": 1, "learning_rate": 1e-3,
    }

    train_hat, test_hat = forecast(train(hyperparameter, X, y, 1440), X, X.iloc[:20], y, 1440)
    train_hat_again, _ = forecast(train(hyperparameter, X, y, 1440), X, X.iloc[:20], y, 1440)

    assert train_hat.shape == (n_rows,) and test_hat.shape == (20,)
    assert np.isfinite(train_hat).all() and np.isfinite(test_hat).all()
    assert np.allclose(train_hat, train_hat_again), "same seed must give the same forecast"

