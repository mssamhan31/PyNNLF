#!/usr/bin/env python
# coding: utf-8

"""Template net load forecasting model: copy this file to add your own.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: implement train_model_<name> to return a fitted object and produce_forecast_<name>
           to return the forecast series; see the add a model documentation.
"""

import pandas as pd

def train_model_mXX_template(hyperparameter, train_df_X, train_df_y, forecast_horizon=None):
    """
    Train the model and return a model object.

    Notes:
        - Your model file name MUST start with an ID prefix: m19_*.py
        - Your function names MUST match the file stem (without .py).
          Example file: m19_my_model.py
          Functions must be:
            - train_model_m19_my_model(...)
            - produce_forecast_m19_my_model(...)

    Args:
        hyperparameter (dict): hyperparameter dict for this run (from models/hyperparameters.yaml)
        train_df_X (pd.DataFrame): predictors for training set
        train_df_y (pd.DataFrame): target for training set, with column 'y'
        forecast_horizon (int | None): forecast horizon in minutes (optional)

    Returns:
        object: model object (can be dict/class) used later in produce_forecast_*
    """
    # Example: no training, just store a bias from hyperparameters
    bias = float(hyperparameter.get("bias", 0.0))
    model = {"bias": bias}
    return model


def produce_forecast_mXX_template(model, train_df_X, test_df_X, train_df_y=None, forecast_horizon=None):
    """
    Produce forecasts for train and test sets.

    Important:
        - Return TWO outputs: (train_y_hat, test_y_hat)
        - Each output can be:
            - pd.Series indexed like train_df_X/test_df_X
            - OR a 1D numpy array of matching length
          (The engine will align/convert to Series.)

    Args:
        model (object): model object returned by train_model_*
        train_df_X (pd.DataFrame): predictors for training set
        test_df_X (pd.DataFrame): predictors for test set
        train_df_y (pd.DataFrame | None): target training set (optional; used by Prophet-like models)
        forecast_horizon (int | None): minutes (optional)

    Returns:
        tuple:
            train_y_hat (pd.Series | np.ndarray): forecast for train
            test_y_hat (pd.Series | np.ndarray): forecast for test
    """
    # Example baseline: last-observation “naive” + bias
    if forecast_horizon is None:
        raise ValueError("forecast_horizon must be provided for this template.")

    horizon = pd.Timedelta(minutes=int(forecast_horizon))
    last_obs_col = f"y_lag_{horizon}m"
    bias = float(model.get("bias", 0.0))

    train_y_hat = train_df_X[last_obs_col] + bias
    test_y_hat = test_df_X[last_obs_col] + bias

    return train_y_hat, test_y_hat