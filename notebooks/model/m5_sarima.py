#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORT IMPORTANT LIBRARY
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


def train_model_m5_sarima(hyperparameter, train_df_X, train_df_y, forecast_horizon):
    ''' Train and test a linear model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training
        forecast_horizon (int) : forecast horizon in mins

    
    Returns:
        model (model) : trained model with all features
    '''
    
    #UNPACK HYPERPARAMETER
    p = hyperparameter['p']
    d = hyperparameter['d']
    q = hyperparameter['q']
    P = hyperparameter['P']
    D = hyperparameter['D']
    Q = hyperparameter['Q']
    seasonal_period_days = hyperparameter['seasonal_period_days']
    
    
    # UPDATE train_df_y to exclude all rows after a sudden jump in the timestep
    train_df_y_updated = remove_jump_df(train_df_y)
        
    # TRAIN MODEL
    # Calculate the frequency of the timesteps using the first and second index values
    timestep_frequency = train_df_y_updated.index[1] - train_df_y_updated.index[0]
    s = int(seasonal_period_days * 24 * 60 / (timestep_frequency.seconds / 60))
    inferred_frequency = pd.infer_freq(train_df_y_updated.index)
    train_df_y_updated = train_df_y_updated.asfreq(inferred_frequency)
    
    # INTRODUCE GAP BETWEEN TRAIN AND TEST SET TO AVOID DATA LEAKAGE
    n_timestep_forecast_horizon = int(forecast_horizon / (timestep_frequency.total_seconds() / 60))
    if n_timestep_forecast_horizon == 1:
        pass
    else:
        train_df_y_updated = train_df_y_updated[:-(n_timestep_forecast_horizon - 1)]
    
    # Assuming train_df_y_updated is your dataframe and 'y' is the column with the training series
    y = train_df_y_updated['y']
    
    # Build and fit the state-space ARIMA model
    model_fitted = SARIMAX(y, order=(p, d, q), seasonal_order = (P, D, Q, s), freq=inferred_frequency).fit()
    
    # PACK MODEL
    model = {"model_fitted": model_fitted}
  

    return model


# In[1]:


def produce_forecast_m5_sarima(model, train_df_X, test_df_X, forecast_horizon):
    """Create forecast at the train and test set using the trained model

    Args:
        model (dictionary): all parameters of the trained model
        train_df_X (df): predictors of train set
        test_df_X (df): predictors of test set
        forecast_horizon (int): forecast horizon in mins

    Returns:
        train_df_y_hat (df) : forecast result at train set
        test_df_y_hat (df) : forecast result at test set
        
    """
    timestep_frequency = test_df_X.index[1] - test_df_X.index[0]
    n_timestep_forecast_horizon = int(forecast_horizon / (timestep_frequency.total_seconds() / 60))
    
    train_df_X_updated = remove_jump_df(train_df_X)
    test_df_X_updated = remove_jump_df(test_df_X)
    
    # UNPACK MODEL
    model_fitted = model['model_fitted']
    
    # PRODUCE FORECAST FOR TRAIN SET
    train_df_y_hat = pd.DataFrame(model_fitted.fittedvalues)
    train_df_y_hat.columns = ['y']

    # train_df_y_hat_2 = pd.DataFrame(model_fitted.forecast(n_timestep_forecast_horizon-1))
    # train_df_y_hat_2.columns = ['y']
    # train_df_y_hat = pd.concat([train_df_y_hat, train_df_y_hat_2])

    train_df_y_hat.index.name = 'datetime'
    
    # TRANSFORM test_df_X to a series with only the last lag
    horizon_timedelta = pd.Timedelta(minutes=forecast_horizon)
    last_observation = f'y_lag_{horizon_timedelta}m'
    test_df_y_last = test_df_X[last_observation]
    
      
    # REFIT THE MODEL AND PRODUCE NEW FORECAST FOR TEST SET
    # THIS CODE RESULTS IN 2 MINS
    test_df_y_hat = pd.DataFrame(index = test_df_X.index)
    test_df_y_hat['y_hat'] = np.nan
    
    # in the case of CV 10, which is when test df < train df
    # don't compute the test forecast
    if (test_df_X.index[-1] < train_df_X.index[0]):
    # this is the case when we use CV10, where the test set is before the train set
        print("Test set is before train set / CV 10, no test forecast can be made")
        return train_df_y_hat, test_df_y_hat

    for i in range(len(test_df_y_last)):
    # for i in range(2): #for test only
        print('Processing i = ', i + 1, ' out of ', len(test_df_y_last)),
        if i == 0:
            test_df_y_hat.iloc[i, 0] = model_fitted.forecast(steps=n_timestep_forecast_horizon).iloc[-1]
        else:
            new_row = pd.DataFrame([test_df_y_last.values[i]], columns=['y'], index=[test_df_y_last.index[i] - dt.timedelta(minutes=forecast_horizon)])
            new_row = new_row.asfreq(test_df_X_updated.index.freq)

            model_fitted = model_fitted.append(new_row)
            test_df_y_hat.iloc[i, 0] = model_fitted.forecast(steps=n_timestep_forecast_horizon).iloc[-1] # to update based on the forecast horizon


    # test_df_y_hat = m06_lr.predict(test_df_X)
    
    return train_df_y_hat, test_df_y_hat

