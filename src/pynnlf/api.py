from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Union
from .runner import run_single, run_batch

PathLike = Union[str, Path]
TestMode = Literal["smoke", "full"]


def run_experiment(spec_path: str | Path) -> None:
    """Run a single experiment from <workspace>/specs/experiment.json."""
    run_single(spec_path)

def run_experiment_batch(spec_path: str | Path) -> None:
    """Run batch experiments from <workspace>/specs/batch.json."""
    run_batch(spec_path)

def run_tests(spec_path: Optional[PathLike] = None, mode: TestMode = "smoke") -> None:
    """
    Run automated tests.
    - smoke: ds0 + fh1 + m6 + hp1 quick check
    - full: run all 18 models and compare against benchmark(s) using a1_experiment_result.csv only
    Outputs go to experiment_result/Archive/Testing Result/ (same as current).
    Chunk 5 will implement full benchmarking logic.
    """
    raise NotImplementedError("Implemented in Chunk 5: smoke + full benchmark tests.")