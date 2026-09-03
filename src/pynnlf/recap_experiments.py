"""Aggregate many experiment result folders into one summary table.

Inputs:  a results root directory containing E#####_* experiment folders.
Outputs: a single CSV holding the first row of each experiment's a1 result, optionally
         also returned as a DataFrame.
Key steps: scan for experiment folders, read each a1 result file, concatenate them, and
           write the combined table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import re

import pandas as pd

PathLike = Union[str, Path]


_EXPERIMENT_DIR_RE = re.compile(r"^E(\d{5})_")


def _coerce_root(path: PathLike) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root


def discover_experiment_folders(results_root: Path) -> list[Path]:
    """
    Discover experiment folders under results_root sorted by experiment index.

    Args:
        results_root (Path): root folder containing experiment directories

    Returns:
        list[Path]: sorted experiment folder paths
    """
    if not results_root.exists():
        return []

    matches: list[tuple[int, str, Path]] = []
    for p in results_root.iterdir():
        if not p.is_dir():
            continue
        m = _EXPERIMENT_DIR_RE.match(p.name)
        if not m:
            continue
        matches.append((int(m.group(1)), p.name, p))

    matches.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in matches]


def find_result_csv(exp_folder: Path) -> Path | None:
    """
    Find a result CSV inside an experiment folder.

    Args:
        exp_folder (Path): experiment directory path

    Returns:
        Path | None: first matching CSV or None if not found
    """
    matches = sorted(exp_folder.glob("*_experiment_result*.csv"))
    if not matches:
        return None
    return matches[0]


def recap_experiments(
    results_root: PathLike = "experiment_result",
    output_path: PathLike | None = None,
    *,
    return_df: bool = False,
    include_experiment_folder: bool = True,
) -> Path | pd.DataFrame:
    """
    Aggregate experiment result CSVs into a single summary CSV.

    Args:
        results_root (str | Path): root folder containing experiment folders
        output_path (str | Path | None): output CSV path (default: <results_root>/a1_experiment_result.csv)
        return_df (bool): if True, return the DataFrame instead of the output path
        include_experiment_folder (bool): add experiment_folder column to output

    Returns:
        Path | pd.DataFrame: output path (default) or DataFrame if return_df=True
    """
    root = _coerce_root(results_root)
    exp_folders = discover_experiment_folders(root)

    rows: list[pd.DataFrame] = []

    for folder in exp_folders:
        csv_path = find_result_csv(folder)
        if csv_path is None:
            print(f"[PyNNLF recap] WARNING: no result file in {folder}")
            continue
        try:
            row_df = pd.read_csv(csv_path, nrows=1)
        except Exception as exc:
            print(f"[PyNNLF recap] WARNING: error reading {csv_path}: {exc}")
            continue

        if include_experiment_folder:
            row_df.insert(0, "experiment_folder", folder.name)

        rows.append(row_df)

    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        cols = ["experiment_folder"] if include_experiment_folder else []
        df = pd.DataFrame(columns=cols)

    out_path = Path(output_path) if output_path is not None else root / "a1_experiment_result.csv"
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df if return_df else out_path
