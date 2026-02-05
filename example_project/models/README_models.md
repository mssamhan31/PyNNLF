# Models folder (auto-discovery)

This folder is **user-editable**. You can add new forecasting models here without editing any package code or config mappings.

## Add a new model (no config edits)

1) Create a new Python file in this folder named like:

- `m19_my_model.py` (example)

2) The file name **must start** with the model ID prefix (`m19_`).

3) Inside that file, define **two functions** whose names match the **file stem** (file name without `.py`).

Example file: `m19_my_model.py`

Required functions:

- `train_model_m19_my_model(hyperparameter, train_df_X, train_df_y, forecast_horizon=None)`
- `produce_forecast_m19_my_model(model, train_df_X, test_df_X, train_df_y=None, forecast_horizon=None)`

## Add hyperparameters (YAML)

Hyperparameters live in:

- `hyperparameters.yaml`

The top-level key **must equal** the model file stem.

Example:

```yaml
m19_my_model:
  hp1:
    bias: 0.0
  hp2:
    bias: 5.0
```

## Run the model (4-key spec only)

Edit `../specs/experiment.yaml`:

```yaml
dataset: ds0
forecast_horizon: fh1
model: m19
hyperparameter: hp1
```

Then run:

```bash
python -c "import pynnlf; pynnlf.run_experiment('PATH_TO_WORKSPACE/specs/experiment.yaml')"
```

## Rules / gotchas

- Auto-discovery requires **exactly one** match for a model ID. If you have both `m19_a.py` and `m19_b.py`, the runner will raise an error asking you to rename.
- Your model outputs can be `pd.Series` or 1D `np.ndarray`. The engine will align outputs to timestamps.
- Keep model files import-safe: include all required imports inside your model file.
