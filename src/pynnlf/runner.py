#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

from .yamlio import load_yaml
from .hyperparams import load_hyperparameters, get_hp
from .discovery import discover_model_name, discover_dataset_path
from .engine import run_experiment_engine

def _workspace_root_from_spec(spec_path: Path) -> Path:
    """
    Infer workspace root from a spec path.

    Args:
        spec_path (Path): <workspace>/specs/experiment.yaml or batch.yaml

    Returns:
        Path: workspace root
    """
    return spec_path.parent.parent

def run_single(spec_path: str | Path) -> None:
    """
    Run a single experiment from a 4-key YAML spec.

    Args:
        spec_path (str | Path): <workspace>/specs/experiment.yaml

    Returns:
        None
    """
    spec_path = Path(spec_path)
    ws = _workspace_root_from_spec(spec_path)

    spec = load_yaml(spec_path)
    cfg = load_yaml(ws / "specs" / "pynnlf_config.yaml")

    ds_id = spec["dataset"]              # e.g. ds19
    fh_id = spec["forecast_horizon"]     # e.g. fh1
    m_id  = spec["model"]                # e.g. m19
    hp_no = spec["hyperparameter"]       # e.g. hp1

    data_dir = ws / cfg["paths"]["data_dir"]
    out_dir  = ws / cfg["paths"]["output_dir"]
    models_dir = ws / "models"
    hp_path = ws / cfg["paths"]["hyperparameters_path"]  # models/hyperparameters.yaml

    # auto-discovery (no config edits)
    dataset_path = discover_dataset_path(data_dir, ds_id)
    model_name = discover_model_name(models_dir, m_id)

    fh_min = int(cfg["forecast_horizons"][fh_id])

    hparams = load_hyperparameters(hp_path)
    hp = get_hp(hparams, model_name, hp_no)

    run_experiment_engine(
        dataset_path=dataset_path,
        forecast_horizon_min=fh_min,
        model_name=model_name,
        hyperparameter_no=hp_no,
        hyperparameter=hp,
        output_dir=out_dir,
        models_dir=models_dir,
        config=cfg,
    )

def run_batch(spec_path: str | Path) -> None:
    """
    Run batch experiments from YAML batch spec (cartesian product).

    Args:
        spec_path (str | Path): <workspace>/specs/batch.yaml

    Returns:
        None
    """
    spec_path = Path(spec_path)
    ws = _workspace_root_from_spec(spec_path)

    batch = load_yaml(spec_path)
    cfg = load_yaml(ws / "specs" / "pynnlf_config.yaml")

    data_dir = ws / cfg["paths"]["data_dir"]
    out_dir  = ws / cfg["paths"]["output_dir"]
    models_dir = ws / "models"
    hp_path = ws / cfg["paths"]["hyperparameters_path"]

    hparams = load_hyperparameters(hp_path)

    for ds_id in batch["datasets"]:
        dataset_path = discover_dataset_path(data_dir, ds_id)
        for fh_id in batch["forecast_horizons"]:
            fh_min = int(cfg["forecast_horizons"][fh_id])
            for m_id, hp_no in batch["model_and_hp"]:
                model_name = discover_model_name(models_dir, m_id)
                hp = get_hp(hparams, model_name, hp_no)
                run_experiment_engine(
                    dataset_path=dataset_path,
                    forecast_horizon_min=fh_min,
                    model_name=model_name,
                    hyperparameter_no=hp_no,
                    hyperparameter=hp,
                    output_dir=out_dir,
                    models_dir=models_dir,
                    config=cfg,
                )