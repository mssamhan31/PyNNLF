#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train_model_m2_snaive(hyperparameter, train_df_X, train_df_y):
    """Train a seasonal naive model for point forecasting.

    The seasonal naive model does not require training; it forecasts 
    using lagged values based on seasonal periods.

    Args:
        hyperparameter (pd.DataFrame): Hyperparameter values, e.g., number of days for seasonality.
        train_df_X (pd.DataFrame): Features matrix for training.
        train_df_y (pd.DataFrame): Target values for training.

    Returns:
        model (dict): Trained seasonal naive model object containing the lagged column name.
    """
    
    #UNPACK HYPERPARAMETER
    days = hyperparameter['days']
    col_name = f'y_lag_{days} days 00:00:00m'
    
    #TRAIN MODEL
    #no training is required for seasonal naive model
  
    # PACK MODEL
    model = {"col_name": col_name }
  

    return model


# In[ ]:


def produce_forecast_m2_snaive(model, train_df_X, test_df_X):
    """Generate seasonal naive forecasts for training and test sets.

    The seasonal naive model forecasts using the value from the same period 
    in the previous season (e.g., same day in prior week).

    Args:
        model (dict): Trained seasonal naive model containing lagged column info.
        train_df_X (pd.DataFrame): Predictor data for the training set.
        test_df_X (pd.DataFrame): Predictor data for the test set.

    Returns:
        train_df_y_hat (pd.Series): Forecast results for the training set.
        test_df_y_hat (pd.Series): Forecast results for the test set.
    """
    
    # UNPACK MODEL
    col_name = model['col_name']  #this depends on the lag day
      
    # PRODUCE FORECAST
    train_df_y_hat = train_df_X[col_name]
    test_df_y_hat = test_df_X[col_name]
    
    return train_df_y_hat, test_df_y_hat

