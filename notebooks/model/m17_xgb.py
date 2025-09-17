#!/usr/bin/env python
# coding: utf-8

# In[3]:


# IMPORT IMPORTANT LIBRARY
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# In[ ]:


def train_model_m17_xgb(hyperparameter, train_df_X, train_df_y):
    """
    Train an XGBoost regressor for point forecasting.

    Args:
        hyperparameter (dict): Dictionary containing model hyperparameters (e.g., n_estimators, learning_rate, max_depth, subsample, colsample_bytree, xgb_seed).
        train_df_X (pd.DataFrame): Features matrix for training.
        train_df_y (pd.DataFrame): Target vector for training.

    Returns:
        model (dict): Dictionary containing the trained XGBoost model under the key 'xgb'.
    """
    
    #UNPACK HYPERPARAMETER
    xgb_seed = int(hyperparameter["xgb_seed"])
    n_estimators=hyperparameter["n_estimators"]
    learning_rate=hyperparameter["learning_rate"]
    max_depth=hyperparameter["max_depth"]
    subsample=hyperparameter["subsample"]
    colsample_bytree=hyperparameter["colsample_bytree"]
    
    #INITIALIZE AND TRAIN MODEL
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=xgb_seed)   
    xgb.fit(train_df_X, train_df_y)   
  
    # PACK MODEL
    model = {"xgb": xgb}
  

    return model


# In[ ]:


def produce_forecast_m17_xgb(model, train_df_X, test_df_X):
    """
    Generate point forecasts for train and test sets using a trained XGBoost model.

    Args:
        model (dict): Dictionary containing the trained XGBoost model under the key 'xgb'.
        train_df_X (pd.DataFrame): Feature matrix for the training set.
        test_df_X (pd.DataFrame): Feature matrix for the test set.

    Returns:
        train_df_y_hat (np.ndarray): Predicted values for the training set.
        test_df_y_hat (np.ndarray): Predicted values for the test set.
    """
    
    # UNPACK MODEL
    xgb = model["xgb"]
    
    # PRODUCE FORECAST
    train_df_y_hat = xgb.predict(train_df_X)
    test_df_y_hat = xgb.predict(test_df_X)
    
    return train_df_y_hat, test_df_y_hat

