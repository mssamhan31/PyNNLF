# Data folder (auto-discovery)

This folder is **user-editable**. You can add new datasets here without editing any package code or config mappings.

## Add a new dataset (no config edits)

1) Drop your CSV into this folder.

2) The file name **must start** with the dataset ID prefix, e.g.:

- `ds19_mydata.csv` (example)

3) Auto-discovery uses the prefix (`ds19`) to find the dataset file.

## Dataset requirements (minimum)

- CSV must have a **datetime index** in the first column (same style as `ds0_test.csv`).
- CSV must contain a target column named `netload_kW` (the engine renames it to `y`).

## Run the dataset (4-key spec only)

Edit `../specs/experiment.yaml`:

```yaml
dataset: ds19
forecast_horizon: fh1
model: m6
hyperparameter: hp1
```

Then run:

```bash
python -c "import pynnlf; pynnlf.run_experiment('PATH_TO_WORKSPACE/specs/experiment.yaml')"
```

## Gotchas

- Auto-discovery requires **exactly one** match for a dataset ID. If you have `ds19_a.csv` and `ds19_b.csv`, the runner will raise an error asking you to rename.
- If your file is huge, keep it out of Git history (use `.gitignore`).
