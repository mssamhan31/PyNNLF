#!/usr/bin/env python
# coding: utf-8

"""Resolve short dataset and model identifiers to files inside a workspace.

Inputs:  a workspace models/ or data/ directory and an identifier such as "m6" or "ds19".
Outputs: the unique matching file path, or the model file stem.
Key steps: glob the directory for "<id>_*<suffix>", fall back to an exact "<id><suffix>"
           name, and raise if there is no match or more than one.
"""

from pathlib import Path

def discover_unique_file(directory: Path, prefix: str, suffix: str):
    """
    Discover a unique file in a directory by prefix and suffix.

    Example:
        prefix="m19" suffix=".py" matches: m19_*.py
        prefix="ds19" suffix=".csv" matches: ds19_*.csv

    Args:
        directory (Path): Directory to search.
        prefix (str): Required filename prefix (e.g. "m19" or "ds19").
        suffix (str): Required suffix (e.g. ".py" or ".csv").

    Returns:
        Path: The unique matching file path.

    Raises:
        FileNotFoundError: If no match found.
        ValueError: If more than one match found.
    """
    directory = Path(directory)
    matches = sorted(directory.glob(f"{prefix}_*{suffix}"))
    if len(matches) == 0:
        # also allow exact name without underscore: ds19.csv / m19.py
        exact = directory / f"{prefix}{suffix}"
        if exact.exists():
            return exact
        raise FileNotFoundError(f"No file found for '{prefix}' in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for '{prefix}' in {directory}: {[m.name for m in matches]}")
    return matches[0]

def discover_model_name(models_dir: Path, model_id: str) -> str:
    """
    Discover model file and return model_name (file stem).

    Args:
        models_dir (Path): <workspace>/models
        model_id (str): model ID from spec, e.g. "m6" or "m19"

    Returns:
        str: model_name file stem, e.g. "m6_lr" or "m19_my_model"
    """
    p = discover_unique_file(models_dir, model_id, ".py")
    return p.stem

def discover_dataset_path(data_dir: Path, dataset_id: str) -> Path:
    """
    Discover dataset file path from dataset ID.

    Args:
        data_dir (Path): <workspace>/data
        dataset_id (str): dataset ID from spec, e.g. "ds0" or "ds19"

    Returns:
        Path: dataset file path.
    """
    return discover_unique_file(data_dir, dataset_id, ".csv")