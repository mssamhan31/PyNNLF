from __future__ import annotations

from pathlib import Path

import pandas as pd

from .recap_experiments import recap_experiments as _recap_experiments
from .runner import run_batch, run_single
from .tests_runner import run_tests as _run_tests


def run_experiment(spec_path: str | Path, *, plot_enabled: bool | None = None) -> None:
    """Run a single experiment from <workspace>/specs/experiment.yaml."""
    run_single(spec_path, plot_enabled=plot_enabled)


def run_experiment_batch(spec_path: str | Path, *, plot_enabled: bool | None = None) -> None:
    """Run batch experiments from <workspace>/specs/batch.yaml."""
    run_batch(spec_path, plot_enabled=plot_enabled)


def run_tests(spec_path: str | Path, *, plot_enabled: bool | None = None) -> Path:
    """Run regression tests from <workspace>/specs/tests_*.yaml and write a report."""
    return _run_tests(spec_path, plot_enabled=plot_enabled)


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
