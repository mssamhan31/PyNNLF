#!/usr/bin/env python
# coding: utf-8

"""Shared model helpers, plus the superseded JSON specification workflow.

Inputs:  forecasting DataFrames for the helpers; JSON specification paths for the
         legacy runners.
Outputs: cleaned DataFrames and separated feature sets; the legacy runners write
         experiment results.
Key steps: remove_jump_df and separate_lag_and_exogenous_features are used by ten of
           the bundled models; the sequence helpers below serve m19_nhits and m20_patchtst. run_single and run_batch are the pre-YAML workflow, kept
           for backward compatibility and superseded by runner.py.
"""

from dataclasses import dataclass
from pathlib import Path
import json

from .hyperparams import load_hyperparameters, get_hp
from .engine import run_experiment_engine

def _load_json(path: Path) -> dict:
    """
    Load JSON file into dict (legacy JSON workflow).

    Args:
        path (Path): JSON file path.

    Returns:
        dict: Parsed JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))

def _workspace_root_from_spec(spec_path: Path) -> Path:
    """
    Infer workspace root from a spec path (legacy JSON workflow).

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
    Run a single experiment from a 4-key JSON spec (legacy workflow).

    Spec contains only:
        dataset, forecast_horizon, model, hyperparameter

    Args:
        spec_path (str | Path): Path to <workspace>/specs/experiment.json

    Returns:
        None

    Deprecated: superseded by the YAML workflow in runner.py, which pynnlf.run_experiment
    and pynnlf.run_experiment_batch call. Kept so existing JSON specifications keep
    working; prefer the YAML entry points for new work.
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
    Run batch experiments from a batch JSON spec (legacy workflow).

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

    Deprecated: superseded by the YAML workflow in runner.py, which pynnlf.run_experiment
    and pynnlf.run_experiment_batch call. Kept so existing JSON specifications keep
    working; prefer the YAML entry points for new work.
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

def separate_lag_and_exogenous_features(train_df_X, target_column='y', lag_prefix='y_lag'):
    '''
    This function separates the lag features and exogenous variables from the training dataframe.

    Args:
        train_df_X (pd.DataFrame): The dataframe containing both lag features and exogenous variables.
        target_column (str): The name of the target column (e.g., 'y').
        lag_prefix (str): The prefix used for lag columns (e.g., 'y_lag').

    Returns:
        X_lags (pd.DataFrame): DataFrame containing only the lag features.
        X_exog (pd.DataFrame): DataFrame containing only the exogenous variables.
    '''
    
    # Identify lag features (columns that start with 'y_lag')
    lag_features = [col for col in train_df_X.columns if col.startswith(lag_prefix)]
    
    # Identify exogenous variables (everything except the target and lag features)
    exog_features = [col for col in train_df_X.columns if col not in [target_column] + lag_features]
    
    # Create dataframes for lag features and exogenous features
    X_lags = train_df_X[lag_features]
    X_exog = train_df_X[exog_features]
    
    return X_lags, X_exog


# ---------------------------------------------------------------------------
# Sequence-model helpers (m19_nhits, m20_patchtst)
# ---------------------------------------------------------------------------
#
# Both models read the same tabular lag matrix the engine builds for every model,
# but as a chronological window rather than a bag of columns. Torch is imported
# inside these helpers so that importing this module stays cheap for the models
# that do not need it.


@dataclass
class SequenceInputs:
    """Tensors a sequence model trains on, plus the statistics to undo the scaling.

    Attributes:
        lookback (torch.Tensor): (n, L) lag windows in chronological order, each
            normalised by its own mean and standard deviation.
        exog (torch.Tensor): (n, n_exog) exogenous features standardised with the
            training-set statistics.
        window_mean (torch.Tensor): (n, 1) per-window mean, in kilowatts (kW).
        window_std (torch.Tensor): (n, 1) per-window standard deviation, in kW.
        target (torch.Tensor | None): (n, 1) target in the window's normalised
            units; None at forecast time.
        exog_mean (np.ndarray): per-column mean used to standardise ``exog``.
        exog_std (np.ndarray): per-column standard deviation used for ``exog``.
    """

    lookback: object
    exog: object
    window_mean: object
    window_std: object
    target: object
    exog_mean: object
    exog_std: object


def prepare_sequence_inputs(df_X, df_y=None, exog_mean=None, exog_std=None):
    """Turn the engine's feature frame into the tensors a sequence model consumes.

    The lag block arrives newest-first, so it is reversed into chronological
    order. Each window is then normalised by its own mean and standard deviation
    (instance normalisation), which lets one network serve datasets of very
    different scale; the forecast is mapped back with the same statistics. The
    exogenous block is standardised with training statistics, computed here when
    none are given so that training and forecasting share them.

    Args:
        df_X (pd.DataFrame): feature frame from the engine.
        df_y (pd.DataFrame | None): target frame; given at training time only.
        exog_mean (np.ndarray | None): training exogenous mean, at forecast time.
        exog_std (np.ndarray | None): training exogenous std, at forecast time.

    Returns:
        SequenceInputs: the prepared tensors and scaling statistics.
    """
    import numpy as np
    import torch

    X_lags, X_exog = separate_lag_and_exogenous_features(df_X)
    lookback = np.ascontiguousarray(X_lags.values[:, ::-1], dtype=np.float32)
    window_mean = lookback.mean(axis=1, keepdims=True)
    window_std = lookback.std(axis=1, keepdims=True) + 1e-5  # guard flat windows
    lookback = (lookback - window_mean) / window_std

    exog = np.asarray(X_exog.values, dtype=np.float32)
    if exog_mean is None:
        exog_mean = exog.mean(axis=0)
        exog_std = exog.std(axis=0)
        exog_std[exog_std == 0] = 1.0  # calendar dummies that never vary in training
    exog = (exog - exog_mean) / exog_std

    target = None
    if df_y is not None:
        target = np.asarray(df_y.values, dtype=np.float32).reshape(-1, 1)
        target = (target - window_mean) / window_std

    return SequenceInputs(
        lookback=torch.from_numpy(lookback),
        exog=torch.from_numpy(exog),
        window_mean=torch.from_numpy(window_mean),
        window_std=torch.from_numpy(window_std),
        target=None if target is None else torch.from_numpy(target),
        exog_mean=exog_mean,
        exog_std=exog_std,
    )


def train_in_minibatches(net, inputs, batch_size, epochs, learning_rate):
    """Train a sequence network with AdamW on mean squared error.

    Prints one line per epoch with the mean training loss and elapsed time so a
    long run can be followed from the console.

    Args:
        net (torch.nn.Module): network taking (lookback, exog) and returning (n, 1).
        inputs (SequenceInputs): training tensors with ``target`` set.
        batch_size (int): minibatch size.
        epochs (int): passes over the training set.
        learning_rate (float): AdamW learning rate.

    Returns:
        None. The network is trained in place.
    """
    import time

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(inputs.lookback, inputs.exog, inputs.target),
        batch_size=batch_size,
        shuffle=True,
    )
    net.train()
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for lookback, exog, target in loader:
            optimiser.zero_grad()
            loss = criterion(net(lookback, exog), target)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(target)
        print(
            f"Epoch [{epoch + 1}/{epochs}], loss: {running_loss / len(loader.dataset):.4f}, "
            f"time taken: {time.time() - start_time:.2f} seconds"
        )


def forecast_in_minibatches(model, df_X, batch_size):
    """Forecast net load from a trained sequence model.

    Args:
        model (dict): mapping with ``net``, ``exog_mean`` and ``exog_std`` as
            returned by the model's train function.
        df_X (pd.DataFrame): feature frame to forecast from.
        batch_size (int): minibatch size, to bound memory on long frames.

    Returns:
        np.ndarray: 1-D forecast in kilowatts (kW), one value per row of ``df_X``.
    """
    import torch

    inputs = prepare_sequence_inputs(df_X, exog_mean=model["exog_mean"], exog_std=model["exog_std"])
    net = model["net"]
    net.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs.lookback), batch_size):
            stop = start + batch_size
            outputs.append(net(inputs.lookback[start:stop], inputs.exog[start:stop]))
    y_hat = torch.cat(outputs) * inputs.window_std + inputs.window_mean
    return y_hat.numpy().ravel()
