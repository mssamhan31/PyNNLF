from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Union
import pandas as pd
from .runner import run_single, run_batch
from .tests_runner import run_tests
from .recap_experiments import recap_experiments as _recap_experiments

PathLike = Union[str, Path]
TestMode = Literal["smoke", "full"]


def run_experiment(spec_path: str | Path) -> None:
    """Run a single experiment from <workspace>/specs/experiment.yaml."""
    run_single(spec_path)

def run_experiment_batch(spec_path: str | Path) -> None:
    """Run batch experiments from <workspace>/specs/batch.yaml."""
    run_batch(spec_path)


def recap_experiments(
    results_root: str | Path = "experiment_result",
    output_path: str | Path | None = None,
    *,
    return_df: bool = False,
    include_experiment_folder: bool = True,
) -> Path | pd.DataFrame:
    """Aggregate experiment summaries into a single CSV."""
    return _recap_experiments(
        results_root=results_root,
        output_path=output_path,
        return_df=return_df,
        include_experiment_folder=include_experiment_folder,
    )