#!/usr/bin/env python
# coding: utf-8

# In[3]:


# IMPORT IMPORTANT LIBRARY
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# In[ ]:


def train_model_m17_xgb(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a xgb model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
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
    """Create forecast at the train and test set using the trained model

    Args:
        model (dictionary): all parameters of the trained model
        train_df_X (df): predictors of train set
        test_df_X (df): predictors of test set

    Returns:
        train_df_y_hat (df) : forecast result at train set
        test_df_y_hat (df) : forecast result at test set
        
    """
    
    # UNPACK MODEL
    xgb = model["xgb"]
    
    # PRODUCE FORECAST
    train_df_y_hat = xgb.predict(train_df_X)
    test_df_y_hat = xgb.predict(test_df_X)
    
    return train_df_y_hat, test_df_y_hat

