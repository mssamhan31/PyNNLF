#!/usr/bin/env python
# coding: utf-8

"""Load YAML files with existence and type checking.

Inputs:  a path to a .yaml or .yml file.
Outputs: the parsed mapping, or an empty mapping for an empty file.
Key steps: confirm the file exists, parse it with yaml.safe_load, and confirm the parsed
           root is a mapping rather than a list or scalar.
"""

from pathlib import Path
from typing import Any, Union
import yaml

PathLike = Union[str, Path]

def load_yaml(path: PathLike) -> dict[str, Any]:
    """
    Load YAML file into a Python dict using safe_load.

    Args:
        path (str | Path): Path to a .yaml/.yml file.

    Returns:
        dict: Parsed YAML content.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping/dict. Got: {type(data)}")
    return data