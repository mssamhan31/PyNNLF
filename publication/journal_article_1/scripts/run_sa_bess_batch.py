from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pynnlf  # noqa: E402


def completed_keys(results_root: Path) -> set[tuple[str, str, str]]:
    """Return completed `(dataset, model_id, hp)` keys from existing experiment outputs."""

    keys: set[tuple[str, str, str]] = set()
    for result_file in sorted(results_root.glob("E*/E*_a1_experiment_result.csv")):
        try:
            row = pd.read_csv(result_file, nrows=1).iloc[0]
        except Exception:
            continue
        dataset_id = str(row.get("dataset_no", ""))
        model_id = str(row.get("model_no", ""))
        hp = str(row.get("hyperparameter_no", ""))
        if dataset_id and model_id and hp:
            keys.add((dataset_id, model_id, hp))
    return keys


def main() -> None:
    """Run missing SA BESS PyNNLF experiments and write the recap CSV.

    The runner is resumable: if a previous long run already produced an
    experiment result CSV for a dataset/model/hyperparameter combination,
    that combination is skipped on the next invocation.
    """

    batch_path = WORKSPACE_DIR / "specs" / "sa_bess_batch.yaml"
    results_root = WORKSPACE_DIR / "experiment_result"
    batch = yaml.safe_load(batch_path.read_text(encoding="utf-8"))
    done = completed_keys(results_root)
    temp_path = WORKSPACE_DIR / "specs" / "_tmp_sa_bess_single.yaml"

    total = len(batch["datasets"]) * len(batch["forecast_horizons"]) * len(batch["model_and_hp"])
    run_index = 0
    for dataset_id in batch["datasets"]:
        for forecast_horizon in batch["forecast_horizons"]:
            for model_id, hp in batch["model_and_hp"]:
                run_index += 1
                key = (str(dataset_id), str(model_id), str(hp))
                if key in done:
                    print(f"[skip {run_index}/{total}] {dataset_id} {model_id} {hp}")
                    continue

                print(f"[run {run_index}/{total}] {dataset_id} {model_id} {hp}")
                temp_spec = {
                    "datasets": [dataset_id],
                    "forecast_horizons": [forecast_horizon],
                    "model_and_hp": [[model_id, hp]],
                }
                temp_path.write_text(yaml.safe_dump(temp_spec, sort_keys=False), encoding="utf-8")
                pynnlf.run_experiment_batch(temp_path, plot_enabled=False)
                done.add(key)

    if temp_path.exists():
        temp_path.unlink()

    pynnlf.recap_experiments(results_root)
    print(f"Full SA BESS batch complete. Recap: {results_root / 'a1_experiment_result.csv'}")


if __name__ == "__main__":
    main()
