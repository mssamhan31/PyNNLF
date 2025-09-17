#!/usr/bin/env python
# coding: utf-8

# In[5]:


# IMPORT IMPORTANT LIBRARY
import pandas as pd
from prophet import Prophet


# In[ ]:


def train_model_m16_prophet(hyperparameter, train_df_X, train_df_y, forecast_horizon):
    """
    Train a Prophet model for point forecasting using lag and exogenous features.

    The model captures trend and seasonality components in the target series,
    and optionally includes exogenous regressors. A gap is introduced between
    training and forecast horizon to prevent data leakage.

    Args:
        hyperparameter (dict): Dictionary containing model hyperparameters, including
            seed, seasonality_prior_scale, seasonality_mode, weekly_seasonality,
            daily_seasonality, and growth.
        train_df_X (pd.DataFrame): Feature matrix for training (lag + exogenous features).
        train_df_y (pd.DataFrame): Target variable for training.
        forecast_horizon (int): Forecast horizon to define the gap between training and testing.

    Returns:
        model (dict): Dictionary containing the trained Prophet model, the processed
            target DataFrame, and associated hyperparameters.
    """
    
    #UNPACK HYPERPARAMETER
    seed = hyperparameter["seed"]
    seasonality_prior_scale = hyperparameter["seasonality_prior_scale"]
    seasonality_mode = hyperparameter["seasonality_mode"]
    weekly_seasonality = hyperparameter["weekly_seasonality"]
    daily_seasonality = hyperparameter["daily_seasonality"]
    growth = hyperparameter["growth"]
    
    
    # UPDATE train_df to exclude all rows after a sudden jump in the timestep
    train_df_y_updated = remove_jump_df(train_df_y)
    train_df_X_updated = remove_jump_df(train_df_X)

    # Calculate the frequency of the timesteps using the first and second index values
    timestep_frequency = train_df_y_updated.index[1] - train_df_y_updated.index[0]
    inferred_frequency = pd.infer_freq(train_df_y_updated.index)
    train_df_y_updated = train_df_y_updated.asfreq(inferred_frequency) 

    # INTRODUCE GAP BETWEEN TRAIN AND TEST SET TO AVOID DATA LEAKAGE
    n_timestep_forecast_horizon = int(forecast_horizon / (timestep_frequency.total_seconds() / 60))
    if n_timestep_forecast_horizon == 1:
        pass
    else:
        train_df_y_updated = train_df_y_updated[:-(n_timestep_forecast_horizon - 1)]
        train_df_X_updated = train_df_X_updated[:-(n_timestep_forecast_horizon - 1)]

    # Assuming train_df_y_updated is your dataframe and 'y' is the column with the training series
    y = train_df_y_updated.copy()
    X_lags, X_exog = separate_lag_and_exogenous_features(train_df_X_updated)

    #Initialize the Prophet model with hyperparameters
    prophet_model = Prophet(
        seasonality_prior_scale=seasonality_prior_scale,  # Example hyperparameter for seasonality strength
        seasonality_mode=seasonality_mode,  # Use multiplicative seasonality
        weekly_seasonality=weekly_seasonality,  # Enable weekly seasonality
        daily_seasonality=daily_seasonality,  # Enable daily seasonality
        growth=growth  # Choose between 'linear' or 'logistic' growth
        # random_state =  seed,  # cannot set seed in prophet
    )
    for col in X_exog.columns:
        prophet_model.add_regressor(col)

    # Add exogenous features to the y DataFrame
    y = y.merge(X_exog, on='datetime')
    y.reset_index(inplace=True)
    y.rename(columns={'datetime': 'ds'}, inplace=True)
    
    # Train model
    prophet_model.fit(y)
  
    # PACK MODEL
    model = {"prophet": prophet_model, "y": y, "hyperparameter": hyperparameter}
  

    return model


# In[ ]:


def produce_forecast_m16_prophet(model, train_df_X, test_df_X, train_df_y, forecast_horizon):
    """
    Generate forecasts for train and test sets using a trained Prophet model, with warm-start updates for test set.

    Args:
        model (dict): Dictionary containing the trained Prophet model, training data, and hyperparameters.
        train_df_X (pd.DataFrame): Predictors of the training set (lag + exogenous features).
        test_df_X (pd.DataFrame): Predictors of the test set (lag + exogenous features).
        train_df_y (pd.DataFrame): Target series of the training set.
        forecast_horizon (int): Forecast horizon in minutes.

    Returns:
        train_df_y_hat (pd.DataFrame): Forecast results for the training set, indexed by datetime.
        test_df_y_hat (pd.DataFrame): Forecast results for the test set, indexed by datetime.
    """
    
    # UNPACK MODEL
    prophet_model = model['prophet']
    y = model['y']
    hyperparameter = model['hyperparameter']
    
    #UNPACK HYPERPARAMETER
    seasonality_prior_scale = hyperparameter["seasonality_prior_scale"]
    seasonality_mode = hyperparameter["seasonality_mode"]
    weekly_seasonality = hyperparameter["weekly_seasonality"]
    daily_seasonality = hyperparameter["daily_seasonality"]
    growth = hyperparameter["growth"]
    
    # Set up X_exog which is used for prediction
    timestep_frequency = test_df_X.index[1] - test_df_X.index[0]
    n_timestep_forecast_horizon = int(forecast_horizon / (timestep_frequency.total_seconds() / 60))
            
    train_df_X_updated = remove_jump_df(train_df_X)
    test_df_X_updated = remove_jump_df(test_df_X)

    X_lags, X_exog = separate_lag_and_exogenous_features(train_df_X_updated)

    X_exog.reset_index(inplace=True)
    X_exog.rename(columns={'datetime': 'ds'}, inplace=True)

    # Forecast train set
    train_df_y_hat = prophet_model.predict(X_exog)

    train_df_y_hat = train_df_y_hat[['ds', 'yhat']]

    train_df_y_hat.set_index('ds', inplace=True)
    train_df_y_hat.index.name = 'datetime'
    
    # Set up function to warm start the model for updating the fit
    def warm_start_params(m):
        """
        Retrieve parameters from a trained model in the format used to initialize a new Stan model.
        Note that the new Stan model must have these same settings:
            n_changepoints, seasonality features, mcmc sampling
        for the retrieved parameters to be valid for the new model.

        Parameters
        ----------
        m: A trained model of the Prophet class.

        Returns
        -------
        A Dictionary containing retrieved parameters of m.
        """
        res = {}
        for pname in ['k', 'm', 'sigma_obs']:
            if m.mcmc_samples == 0:
                res[pname] = m.params[pname][0][0]
            else:
                res[pname] = np.mean(m.params[pname])
        for pname in ['delta', 'beta']:
            if m.mcmc_samples == 0:
                res[pname] = m.params[pname][0]
            else:
                res[pname] = np.mean(m.params[pname], axis=0)
        return res

    # PRODUCE FORECASTFOR TEST SET
    
    # REFIT THE MODEL AND PRODUCE NEW FORECAST FOR TEST SET
    # The model is refitted for 100 times only so there will be only 100 forecast results. 
    
    test_df_y_hat = pd.DataFrame(index = test_df_X.index)
    test_df_y_hat['y_hat'] = np.nan


    # in the case of CV 10, which is when test df < train df
    # don't compute the test forecast
    if (test_df_X.index[-1] < train_df_X.index[0]):
    # this is the case when we use CV10, where the test set is before the train set
        print("Test set is before train set / CV 10, no test forecast can be made")
        return train_df_y_hat, test_df_y_hat
    
    _, X_test = separate_lag_and_exogenous_features(test_df_X)
    X_test.reset_index(inplace=True)
    X_test.rename(columns={'datetime': 'ds'}, inplace=True)

    n_update = 100
    n_timesteps_per_update = int(len(test_df_y_hat) / (n_update + 1))
    
    # TRANSFORM test_df_X to a series with only the last lag
    horizon_timedelta = pd.Timedelta(minutes=forecast_horizon)
    last_observation = f'y_lag_{horizon_timedelta}m'
    test_df_y_last = test_df_X[last_observation]

    new_y = pd.DataFrame(test_df_y_last)
    new_y.rename(columns={new_y.columns[0]: 'y'}, inplace=True)
    new_y.insert(0, 'ds', new_y.index - pd.Timedelta(minutes=forecast_horizon))
    new_y.reset_index(drop = True, inplace=True)

    new_y = new_y.drop(0, axis=0).reset_index(drop=True)
    X_exog_complete = pd.concat([X_exog, X_test], axis=0)
    X_exog_complete = X_exog_complete.drop(0, axis=0).reset_index(drop=True)
    new_y = pd.merge(new_y, X_exog_complete, on='ds', how='left')
    
    for i in range(n_update):
    # for i in range(2): #for test only
        print('Processing i = ', i + 1, ' out of ', n_update),
        if i == 0:
            X_test_curr = X_test.iloc[:1,:]
            test_df_y_hat.iloc[i, 0] = prophet_model.predict(X_test_curr)['yhat'].values[0]
        else:
            new_rows = new_y.iloc[(i-1)*n_timesteps_per_update : i*n_timesteps_per_update, :]
            y = pd.concat([y, new_rows], ignore_index=True)

            current_params = warm_start_params(prophet_model)

            prophet_model = Prophet(
                seasonality_prior_scale=seasonality_prior_scale,  # Example hyperparameter for seasonality strength
                seasonality_mode=seasonality_mode,  # Use multiplicative seasonality
                weekly_seasonality=weekly_seasonality,  # Enable weekly seasonality
                daily_seasonality=daily_seasonality,  # Enable daily seasonality
                growth=growth,  # Choose between 'linear' or 'logistic' growth
            )

            prophet_model = prophet_model.fit(y, init=current_params)  # Adding the last day, warm-starting from the prev model
            X_test_curr = X_test.iloc[i*n_timesteps_per_update : (1+i*n_timesteps_per_update),:]
            test_df_y_hat.iloc[i*n_timesteps_per_update, 0] = prophet_model.predict(X_test_curr)['yhat'].values[0]
    
    return train_df_y_hat, test_df_y_hat

