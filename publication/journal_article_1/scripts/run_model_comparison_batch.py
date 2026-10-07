"""Run the 14-model x 4-dataset comparison for the journal article, resumably.

Inputs:  ``specs/model_comparison_fh8_batch.yaml`` (datasets, the single horizon,
         and the model/hyperparameter pairs in light-to-heavy order), plus any
         ``E*_a1_experiment_result.csv`` already present in ``experiment_result/``
         or ``experiment_result_databricks/``.
Outputs: one experiment folder per missing (dataset, model, hp) under
         ``experiment_result_databricks/`` and a refreshed ``a1_experiment_result.csv``
         recap there.
Key steps: collect the completed keys from both result stores, run each missing
           combination as a one-line batch spec, then recap.

Usage (from ``publication/journal_article_1/``)::

    python scripts/run_model_comparison_batch.py            # run what is missing
    python scripts/run_model_comparison_batch.py --dry-run  # list only

The run-level skip means the script can be restarted after an interruption and
will pick up where it stopped; a combination counts as done only once its a1
summary exists, so a fold-level interruption reruns that combination.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pynnlf  # noqa: E402

BATCH_PATH = WORKSPACE_DIR / "specs" / "model_comparison_fh8_batch.yaml"
OUTPUT_ROOT = WORKSPACE_DIR / "experiment_result_databricks"
RESULT_STORES = [WORKSPACE_DIR / "experiment_result", OUTPUT_ROOT]
TEMP_SPEC_PATH = WORKSPACE_DIR / "specs" / "_tmp_model_comparison_single.yaml"
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


def main(dry_run: bool = False) -> None:
    """Run every missing combination in the batch spec, then recap the output store.

    Args:
        dry_run (bool): when True, print the plan and run nothing.

    Returns:
        None.
    """
    batch = yaml.safe_load(BATCH_PATH.read_text(encoding="utf-8"))
    if batch["forecast_horizons"] != ["fh8"]:
        raise ValueError("This runner keys completed work on the 1-day horizon; the spec must list fh8 only.")
    done = completed_keys(RESULT_STORES)
    OUTPUT_ROOT.mkdir(exist_ok=True)

    plan = [(str(dataset_id), str(model_id), str(hp)) for dataset_id in batch["datasets"] for model_id, hp in batch["model_and_hp"]]
    pending = [key for key in plan if key not in done]
    print(f"{len(plan)} combinations in the spec, {len(plan) - len(pending)} already done, {len(pending)} to run")
    for key in plan:
        print(f"  [{'skip' if key in done else 'run '}] {' '.join(key)}")
    if dry_run or not pending:
        return

    for index, (dataset_id, model_id, hp) in enumerate(pending, start=1):
        print(f"[run {index}/{len(pending)}] {dataset_id} {model_id} {hp}", flush=True)
        temp_spec = {"datasets": [dataset_id], "forecast_horizons": ["fh8"], "model_and_hp": [[model_id, hp]]}
        TEMP_SPEC_PATH.write_text(yaml.safe_dump(temp_spec, sort_keys=False), encoding="utf-8")
        pynnlf.run_experiment_batch(TEMP_SPEC_PATH, plot_enabled=False)

    if TEMP_SPEC_PATH.exists():
        TEMP_SPEC_PATH.unlink()
    pynnlf.recap_experiments(OUTPUT_ROOT)
    print(f"Model comparison batch complete. Recap: {OUTPUT_ROOT / 'a1_experiment_result.csv'}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv[1:])
