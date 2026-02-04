import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]

def load_hyperparameters(path: PathLike) -> dict[str, Any]:
    """
    Load hyperparameters JSON.

    Args:
        path (str | Path): Path to models/hyperparameters.json.

    Returns:
        dict: Nested dict: {model_name: {hp_no: hyperparameter_dict}}.
    """
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))

def get_hp(hparams: dict[str, Any], model_name: str, hp_no: str) -> dict[str, Any]:
    """
    Get hyperparameter dict for a specific model and hp_no.

    Args:
        hparams (dict): Output of load_hyperparameters().
        model_name (str): e.g., "m6_lr".
        hp_no (str): e.g., "hp1".

    Returns:
        dict: Hyperparameter dictionary for the requested model/hp.
    """
    if model_name not in hparams:
        raise KeyError(f"Model '{model_name}' not found in hyperparameters.json")
    if hp_no not in hparams[model_name]:
        raise KeyError(f"HP '{hp_no}' not found for model '{model_name}'")
    hp = hparams[model_name][hp_no]
    if not isinstance(hp, dict):
        raise TypeError("Hyperparameter entry must be a JSON object/dict")
    return hp