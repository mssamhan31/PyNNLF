from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "m17_xgb_hp1",
    "m8_dnn_hp1",
    "m6_lr_hp1",
    "m10_rf_hp1",
    "m7_ann_hp1",
    "m1_naive_hp1",
    "m3_ets_hp1",
    "m9_rt_hp3",
    "m13_lstm_hp2",
    "m2_snaive_hp2",
    "m16_prophet_hp1",
    "m4_arima_hp1",
]

DATASET_LABELS = {
    "ds16": "underlying_load",
    "ds17": "net_load_with_pv",
    "ds18": "net_load_with_pv_battery",
}


def _workspace_path(workspace_dir: str | Path | None = None) -> Path:
    """Resolve the publication workspace path."""

    if workspace_dir is not None:
        return Path(workspace_dir).resolve()
    return Path(__file__).resolve().parents[1]


def load_recap(workspace_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the PyNNLF recap CSV from the publication workspace."""

    workspace = _workspace_path(workspace_dir)
    recap_path = workspace / "experiment_result" / "a1_experiment_result.csv"
    if not recap_path.exists():
        raise FileNotFoundError(
            f"PyNNLF recap not found at {recap_path}. Run pynnlf.recap_experiments first."
        )
    recap = pd.read_csv(recap_path)
    if recap.empty:
        raise ValueError(f"PyNNLF recap is empty: {recap_path}")
    return recap


def _normalise_recap(recap: pd.DataFrame) -> pd.DataFrame:
    """Add stable publication labels to the raw PyNNLF recap table."""

    required = {"dataset_no", "model_name", "test_nRMSE", "test_nRMSE_stddev"}
    missing = sorted(required - set(recap.columns))
    if missing:
        raise ValueError(f"Recap is missing required columns: {missing}")

    result = recap.copy()
    result["dataset_label"] = result["dataset_no"].map(DATASET_LABELS)
    result = result.loc[result["dataset_label"].notna()].copy()

    unknown_models = sorted(set(result["model_name"]) - set(MODEL_ORDER))
    if unknown_models:
        raise ValueError(f"Unexpected model/hyperparameter labels in recap: {unknown_models}")

    result["model_name"] = pd.Categorical(result["model_name"], categories=MODEL_ORDER, ordered=True)
    result["dataset_label"] = pd.Categorical(
        result["dataset_label"],
        categories=[DATASET_LABELS[key] for key in ["ds16", "ds17", "ds18"]],
        ordered=True,
    )
    return result.sort_values(["model_name", "dataset_label"])


def _wide_metric_table(recap: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """Pivot one metric into a model-by-dataset publication table."""

    table = recap.pivot_table(
        index="model_name",
        columns="dataset_label",
        values=metric_column,
        aggfunc="first",
        observed=False,
    )
    table = table.reindex(index=MODEL_ORDER)
    table = table[[DATASET_LABELS[key] for key in ["ds16", "ds17", "ds18"]]]
    table.index.name = "model_hp"
    table.columns.name = None
    return table


def build_publication_tables(workspace_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create nRMSE and nRMSE standard-deviation comparison CSVs."""

    workspace = _workspace_path(workspace_dir)
    results_dir = workspace / "results" / "00_data_exploration_and_processing"
    results_dir.mkdir(parents=True, exist_ok=True)

    recap = _normalise_recap(load_recap(workspace))
    expected_rows = len(MODEL_ORDER) * len(DATASET_LABELS)
    if recap.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} recap rows for SA BESS, found {recap.shape[0]}.")

    nrmse_table = _wide_metric_table(recap, "test_nRMSE")
    nrmse_stddev_table = _wide_metric_table(recap, "test_nRMSE_stddev")

    if nrmse_table.shape != (12, 3):
        raise ValueError(f"nRMSE table has unexpected shape {nrmse_table.shape}; expected (12, 3).")
    if nrmse_stddev_table.shape != (12, 3):
        raise ValueError(
            f"nRMSE stddev table has unexpected shape {nrmse_stddev_table.shape}; expected (12, 3)."
        )

    nrmse_table.to_csv(results_dir / "sa_bess_nrmse_comparison.csv")
    nrmse_stddev_table.to_csv(results_dir / "sa_bess_nrmse_stddev_comparison.csv")
    return nrmse_table, nrmse_stddev_table


# Entry by notebooks/imported helpers only.
