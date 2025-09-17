#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train_model_m1_naive(hyperparameter, train_df_X, train_df_y):
    """Train a naive model for point forecasting.

    For the naive model, no actual training is required. The model 
    simply stores configuration (if any) and will produce lagged forecasts.

    Args:
        hyperparameter (pd.DataFrame): Hyperparameter values (e.g., number of features).
        train_df_X (pd.DataFrame): Features matrix for training.
        train_df_y (pd.DataFrame): Target values for training.

    Returns:
        model (dict): Trained naive model object containing all features.
    """
    
    #UNPACK HYPERPARAMETER
    #no hyperparameter for naive model
    
    #TRAIN MODEL
    #no training is required for naive model
  
    # PACK MODEL
    model = {}
  

    return model


# In[ ]:


def produce_forecast_m1_naive(model, train_df_X, test_df_X):
    """Generate naive forecasts for training and test sets using lagged values.

    Args:
        model (dict): Parameters of the trained model.
        train_df_X (pd.DataFrame): Predictor data for the training set.
        test_df_X (pd.DataFrame): Predictor data for the test set.

    Returns:
        train_df_y_hat (pd.Series): Forecast results for the training set.
        test_df_y_hat (pd.Series): Forecast results for the test set.
    """
    
    # PRODUCE FORECAST
    horizon_timedelta = pd.Timedelta(minutes=forecast_horizon)
    last_observation = f'y_lag_{horizon_timedelta}m'
    train_df_y_hat = train_df_X[last_observation]
    test_df_y_hat = test_df_X[last_observation]
    
    return train_df_y_hat, test_df_y_hat

