"""Run the 14-model x 4-dataset comparison for the journal article, resumably.

Inputs:  ``specs/model_comparison_fh8_batch.yaml`` (datasets, the single horizon,
         and the model/hyperparameter pairs in light-to-heavy order, which the
         runner walks model by model across all datasets), the
         workspace config and hyperparameter file, plus any
         ``E*_a1_experiment_result.csv`` already present in ``experiment_result/``
         or ``experiment_result_databricks/``.
Outputs: one experiment folder per missing (dataset, model, hp) under
         ``experiment_result_databricks/`` and a refreshed ``a1_experiment_result.csv``
         recap there.
Key steps: collect the completed keys from both result stores, run each missing
           combination through the engine with the Databricks output folder and
           model pickling off, prune the training folds, then recap.

Usage (from ``publication/journal_article_1/``)::

    python scripts/run_model_comparison_batch.py            # run what is missing
    python scripts/run_model_comparison_batch.py --dry-run  # list only

The Databricks notebook ``notebooks/01_ashd_acap_edp_148hh_comparison/
0_run_model_comparison_databricks.ipynb`` imports this module and calls
:func:`main`, so both entry points share one implementation.

The run-level skip means the script can be restarted after an interruption and
will pick up where it stopped; a combination counts as done only once its a1
summary exists, so a fold-level interruption reruns that combination.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pynnlf  # noqa: E402
from pynnlf.discovery import discover_dataset_path, discover_model_name  # noqa: E402
from pynnlf.engine import run_experiment_engine  # noqa: E402
from pynnlf.hyperparams import get_hp, load_hyperparameters  # noqa: E402

BATCH_PATH = WORKSPACE_DIR / "specs" / "model_comparison_fh8_batch.yaml"
CONFIG_PATH = WORKSPACE_DIR / "specs" / "pynnlf_config.yaml"
OUTPUT_ROOT = WORKSPACE_DIR / "experiment_result_databricks"
RESULT_STORES = [WORKSPACE_DIR / "experiment_result", OUTPUT_ROOT]
HORIZON_MINUTES = 1440  # fh8; the comparison is run at the 1-day horizon only


def completed_keys(results_roots: list[Path]) -> set[tuple[str, str, str]]:
    """Return the `(dataset, model_id, hp)` keys already run at the 1-day horizon.

    Args:
        results_roots (list[Path]): result stores to scan for a1 summaries.

    Returns:
        set[tuple[str, str, str]]: keys such as ("ds20", "m17", "hp1").
    """
    keys: set[tuple[str, str, str]] = set()
    for root in results_roots:
        for result_file in sorted(root.glob("E*/E*_a1_experiment_result.csv")):
            try:
                row = pd.read_csv(result_file, nrows=1).iloc[0]
            except Exception:
                continue  # a half-written summary is treated as not done
            if pd.to_numeric(row.get("forecast_horizon_min"), errors="coerce") != HORIZON_MINUTES:
                continue
            key = (str(row.get("dataset_no", "")), str(row.get("model_no", "")), str(row.get("hyperparameter_no", "")))
            if all(key):
                keys.add(key)
    return keys


def load_workspace_config() -> dict:
    """Read the workspace config with plotting and model pickling switched off.

    The pickles are never read back, and the LSTM and Prophet ones run to hundreds
    of megabytes per fold, which breaks the 1 GB working-branch limit of a
    Databricks Git folder.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config.setdefault("plot", {})["enabled"] = False
    config.setdefault("output", {})["save_models"] = False
    return config


def prune_training_folds(results_root: Path) -> None:
    """Delete the per-fold training series the engine writes; the a3 file keeps the fold metrics.

    They are ignored by Git, but on Databricks they would still count towards the
    Git folder size limit.

    Args:
        results_root (Path): result store to prune.

    Returns:
        None.
    """
    for folder in results_root.glob("E*/E*_cv_train"):
        shutil.rmtree(folder, ignore_errors=True)


def main(dry_run: bool = False) -> None:
    """Run every missing combination in the batch spec, then recap the output store.

    The engine is called directly with ``OUTPUT_ROOT`` as its output folder; the
    workspace config's own ``output_dir`` (the laptop store) is left untouched.

    Args:
        dry_run (bool): when True, print the plan and run nothing.

    Returns:
        None.
    """
    batch = yaml.safe_load(BATCH_PATH.read_text(encoding="utf-8"))
    config = load_workspace_config()
    if batch["forecast_horizons"] != ["fh8"] or int(config["forecast_horizons"]["fh8"]) != HORIZON_MINUTES:
        raise ValueError("This runner keys completed work on the 1-day horizon; the spec must list fh8 only.")
    done = completed_keys(RESULT_STORES)
    OUTPUT_ROOT.mkdir(exist_ok=True)

    # Model-major order: every dataset for the cheapest model first, the LSTM on
    # every dataset last, so an early stop still leaves whole rows of the table.
    plan = [(str(dataset_id), str(model_id), str(hp)) for model_id, hp in batch["model_and_hp"] for dataset_id in batch["datasets"]]
    pending = [key for key in plan if key not in done]
    print(f"{len(plan)} combinations in the spec, {len(plan) - len(pending)} already done, {len(pending)} to run")
    for key in plan:
        print(f"  [{'skip' if key in done else 'run '}] {' '.join(key)}")
    if dry_run or not pending:
        return

    data_dir = WORKSPACE_DIR / config["paths"]["data_dir"]
    models_dir = WORKSPACE_DIR / "models"
    hyperparameters = load_hyperparameters(WORKSPACE_DIR / config["paths"]["hyperparameters_path"])
    for index, (dataset_id, model_id, hp_no) in enumerate(pending, start=1):
        model_name = discover_model_name(models_dir, model_id)
        print(f"[run {index}/{len(pending)}] {dataset_id} {model_id} {hp_no} -> {model_name}", flush=True)
        run_experiment_engine(
            dataset_path=discover_dataset_path(data_dir, dataset_id),
            forecast_horizon_min=HORIZON_MINUTES,
            model_name=model_name,
            hyperparameter_no=hp_no,
            hyperparameter=get_hp(hyperparameters, model_name, hp_no),
            output_dir=OUTPUT_ROOT,
            models_dir=models_dir,
            config=config,
        )
        prune_training_folds(OUTPUT_ROOT)

    pynnlf.recap_experiments(OUTPUT_ROOT, output_path=OUTPUT_ROOT / "a1_experiment_result.csv")
    print(f"Model comparison batch complete. Recap: {OUTPUT_ROOT / 'a1_experiment_result.csv'}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv[1:])
