#!/usr/bin/env python
# coding: utf-8

"""Load model hyperparameters from a workspace hyperparameters.yaml.

Inputs:  path to a workspace hyperparameters.yaml, a model name and a hyperparameter number.
Outputs: the hyperparameter mapping for that model and number.
Key steps: load the YAML file, then index it by [model_name][hp_no].
"""

from typing import Any
from .yamlio import load_yaml

def load_hyperparameters(path) -> dict[str, Any]:
    """
    Load hyperparameters from YAML.

    Args:
        path (str | Path): Path to models/hyperparameters.yaml.

    Returns:
        dict: {model_name: {hp_no: hyperparameter_dict}}
    """
    return load_yaml(path)

def get_hp(hparams: dict[str, Any], model_name: str, hp_no: str) -> dict[str, Any]:
    """
    Get hyperparameter dict for a specific model and hp_no.

    Args:
        hparams (dict): Output of load_hyperparameters().
        model_name (str): e.g., "m6_lr" (model file stem)
        hp_no (str): e.g., "hp1"

    Returns:
        dict: Hyperparameter dict for the model/hp.
    """
    if model_name not in hparams:
        raise KeyError(f"Model '{model_name}' not found in hyperparameters.yaml")
    if hp_no not in hparams[model_name]:
        raise KeyError(f"HP '{hp_no}' not found for model '{model_name}'")
    hp = hparams[model_name][hp_no]
    if not isinstance(hp, dict):
        raise TypeError("Hyperparameter entry must be a mapping/dict")
    return hp