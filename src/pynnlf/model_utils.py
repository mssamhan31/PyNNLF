#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import json

from .hyperparams import load_hyperparameters, get_hp
from .engine import run_experiment_engine

def _load_json(path: Path) -> dict:
    """
    Load JSON file into dict.

    Args:
        path (Path): JSON file path.

    Returns:
        dict: Parsed JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))

def _workspace_root_from_spec(spec_path: Path) -> Path:
    """
    Infer workspace root from a spec path.

    Assumes structure:
        <workspace>/specs/experiment.json
        <workspace>/specs/batch.json

    Args:
        spec_path (Path): Path to spec JSON.

    Returns:
        Path: Workspace root directory.
    """
    return spec_path.parent.parent

def run_single(spec_path: str | Path) -> None:
    """
    Run a single experiment from a 4-key JSON spec.

    Spec contains only:
        dataset, forecast_horizon, model, hyperparameter

    Args:
        spec_path (str | Path): Path to <workspace>/specs/experiment.json

    Returns:
        None
    """
    spec_path = Path(spec_path)
    ws = _workspace_root_from_spec(spec_path)

    spec = _load_json(spec_path)
    cfg = _load_json(ws / "specs" / "pynnlf_config.json")

    ds_id = spec["dataset"]
    fh_id = spec["forecast_horizon"]
    m_id  = spec["model"]
    hp_no = spec["hyperparameter"]

    data_dir = ws / cfg["paths"]["data_dir"]
    out_dir  = ws / cfg["paths"]["output_dir"]
    hp_path  = ws / cfg["paths"]["hyperparameters_path"]  # models/hyperparameters.json

    dataset_file = cfg["datasets"][ds_id]
    fh_min       = int(cfg["forecast_horizons"][fh_id])
    model_name   = cfg["models"][m_id]

    dataset_path = data_dir / dataset_file
    models_dir   = ws / "models"

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
    Run batch experiments from a batch JSON spec.

    Batch spec contains:
        datasets: [dsX...]
        forecast_horizons: [fhX...]
        model_and_hp: [[mX, hpY], ...]

    Runs all combinations:
        datasets × forecast_horizons × model_and_hp

    Args:
        spec_path (str | Path): Path to <workspace>/specs/batch.json

    Returns:
        None
    """
    spec_path = Path(spec_path)
    ws = _workspace_root_from_spec(spec_path)

    batch = _load_json(spec_path)
    cfg = _load_json(ws / "specs" / "pynnlf_config.json")

    data_dir = ws / cfg["paths"]["data_dir"]
    out_dir  = ws / cfg["paths"]["output_dir"]
    hp_path  = ws / cfg["paths"]["hyperparameters_path"]
    models_dir = ws / "models"

    hparams = load_hyperparameters(hp_path)

    ds_files = [cfg["datasets"][d] for d in batch["datasets"]]
    fh_mins  = [int(cfg["forecast_horizons"][h]) for h in batch["forecast_horizons"]]
    model_and_hp = [(cfg["models"][m], hp) for (m, hp) in batch["model_and_hp"]]

    for ds_file in ds_files:
        for fh_min in fh_mins:
            for model_name, hp_no in model_and_hp:
                dataset_path = data_dir / ds_file
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
                
# transform below scripts into function with input train_df_y and output train_df_y_updated
def remove_jump_df(train_df_y):
    #make docstring with the same format like other cells
    """
    Remove jump in the time series data
    Parameters:
        train_df_y (pd.Series): Time series data
        
    Returns:
        train_df_y_updated (pd.Series): Time series data with jump removed
    """
    
    time_diff = train_df_y.index.to_series().diff().dt.total_seconds()
    initial_freq = time_diff.iloc[1]
    jump_indices = time_diff[time_diff > initial_freq].index
    if not jump_indices.empty:
        jump_index = jump_indices[0]
        jump_pos = train_df_y.index.get_loc(jump_index)
        train_df_y_updated = train_df_y.iloc[:jump_pos]
    else:
        train_df_y_updated = train_df_y
    return train_df_y_updated