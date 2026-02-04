from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Union
from .runner import run_single, run_batch
from .tests_runner import run_tests

PathLike = Union[str, Path]
TestMode = Literal["smoke", "full"]


def run_experiment(spec_path: str | Path) -> None:
    """Run a single experiment from <workspace>/specs/experiment.json."""
    run_single(spec_path)

def run_experiment_batch(spec_path: str | Path) -> None:
    """Run batch experiments from <workspace>/specs/batch.json."""
    run_batch(spec_path)