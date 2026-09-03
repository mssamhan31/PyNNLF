"""Seasonal naive net load forecasting model: the value one season ago.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: select the lag column one seasonal period back and use it as the forecast.
"""

def train_model_m2_snaive(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a seasonal model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
    #UNPACK HYPERPARAMETER
    days = hyperparameter['days']
    col_name = f'y_lag_{days} days 00:00:00m'
    
    #TRAIN MODEL
    #no training is required for seasonal naive model
  
    # PACK MODEL
    model = {"col_name": col_name }
  

    return model

def produce_forecast_m2_snaive(model, train_df_X, test_df_X):
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
    col_name = model['col_name']  #this depends on the lag day
      
    # PRODUCE FORECAST
    train_df_y_hat = train_df_X[col_name]
    test_df_y_hat = test_df_X[col_name]
    
    return train_df_y_hat, test_df_y_hat

