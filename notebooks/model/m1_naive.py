#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train_model_m1_naive(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a naive model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
    #UNPACK HYPERPARAMETER
    #no hyperparameter for naive model
    
    #TRAIN MODEL
    #no training is required for naive model
  
    # PACK MODEL
    model = {}
  

    return model


# In[1]:


def produce_forecast_m1_naive(model, train_df_X, test_df_X):
    """Create forecast at the train and test set using the trained model

    Args:
        model (dictionary): all parameters of the trained model
        train_df_X (df): predictors of train set
        test_df_X (df): predictors of test set

    Returns:
        train_df_y_hat (df) : forecast result at train set
        test_df_y_hat (df) : forecast result at test set
        
    """
    
    # PRODUCE FORECAST
    horizon_timedelta = pd.Timedelta(minutes=forecast_horizon)
    last_observation = f'y_lag_{horizon_timedelta}m'
    train_df_y_hat = train_df_X[last_observation]
    test_df_y_hat = test_df_X[last_observation]
    
    return train_df_y_hat, test_df_y_hat

