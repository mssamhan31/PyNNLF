#!/usr/bin/env python
# coding: utf-8

# # IMPORT ALL NECESSARY LIBRARY

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import os
import time
import pickle # for saving trained model
import dill # for saving trained model


# # FOLDER PREPARATION

# In[ ]:


def compute_exp_no(path_result):
    """Compute experiment number for folder & file naming based on the number of existing experiments that have been done.
    For example, if on the folder there are already 5 experiment folders, then the new experiment no. is E00006.

    Args:
        path_result (str): relative path of experiment folder, stored in config

    Returns:
        int: exp no 
        str: exp no in str
    """
    subfolders = os.listdir(path_result)
    number_of_folders = len(subfolders)
    experiment_no = number_of_folders - 1 # there is one archive folder
    experiment_no_str = f"E{str(experiment_no).zfill(5)}"
    
    return experiment_no, experiment_no_str

def compute_folder_name(experiment_no_str, forecast_horizon, model_name, hyperparameter_no):
    """Folder name in the format of [exp number]_[exp date]_[dataset]_[forecast horizon]_[model]_[hyperparameter]

    Args:
        experiment_no_str (str): exp number
        forecast_horizon (int): forecast horizon in minutes
        model_name (str): for example, m1_naive
        hyperparameter_no (str): for example, hp1

    Returns:
        str: folder name
    """
    folder_name = \
        experiment_no_str + '_' +\
        datetime.today().date().strftime("%y%m%d") + '_' +\
        dataset.split('_')[0] + '_' +\
        'fh' + str(forecast_horizon) + '_' +\
        model_name + '_' +\
        hyperparameter_no
    return folder_name

def prepare_directory(path_result, forecast_horizon, model_name, hyperparameter_no):
    """Do two things,
    1. Create folders inside the experiment result folder
    2. Create some file names to be used when exporting result later
    

    Args:
        path_result (str): relative path to the experiment result folder
        forecast_horizon (int): forecast horizon in minutes
        model_name (str): eg m1_naive
        hyperparameter_no (str): eg

    Returns:
        hyperparameter (df): pd df series of hyperparameter chosen
        experiment_no_str (str) : experiment number
        filepath (dict) : dict of all filepaths that will be exported
    """
    
    hyperparameter_table = globals()[f"{model_name.split('_')[0]}_hp_table"]
    hyperparameter = hyperparameter_table.loc[hyperparameter_no]
    
    experiment_no, experiment_no_str = compute_exp_no(path_result)
    folder_name = compute_folder_name(experiment_no_str, forecast_horizon, model_name, hyperparameter_no)

    # CREATE FOLDER
    cv_folder_train = experiment_no_str + '_cv_train'
    cv_folder_test = experiment_no_str + '_cv_test'
    cv1_plot_folder = experiment_no_str + '_cv1_plots'
    folder_model = experiment_no_str + '_models'
    
    path_result2 = path_result + folder_name +'/'
    path_result_train = path_result2 + cv_folder_train +'/'
    path_result_test = path_result2 + cv_folder_test +'/'
    path_result_plot = path_result2 + cv1_plot_folder +'/'
    path_model = path_result2 + folder_model +'/'

    # MAKE FOLDERS
    os.mkdir(path_result2)
    os.mkdir(path_result_train)
    os.mkdir(path_result_test)
    os.mkdir(path_result_plot)
    os.mkdir(path_model)

    # MAKE FILE PATH
    filepath = {
        'a1' : path_result2 + experiment_no_str + '_a1_experiment_result.csv',
        'a2' : path_result2 + experiment_no_str + '_a2_hyperparameter.csv',
        'a3' : path_result2 + experiment_no_str + '_a3_cross_validation_result.csv',
        'b1' : path_result_plot + experiment_no_str + '_b1_train_timeplot.png', # Time Plot of Forecast vs Observation
        'b2' : path_result_plot + experiment_no_str + '_b2_train_scatterplot.png', # Scatter Plot of Forecast vs Observation
        'b3' : path_result_plot + experiment_no_str + '_b3_train_residual_timeplot.png', # Time Plot of Residual
        'b4' : path_result_plot + experiment_no_str + '_b4_train_residual_histogram.png', # Histogram of Residual
        'b5' : path_result_plot + experiment_no_str + '_b5_train_learningcurve.png', # Learning Curve vs Epoch
        'c1' : path_result_plot + experiment_no_str + '_c1_test_timeplot.png',  # Time Plot of Forecast vs Observation
        'c2' : path_result_plot + experiment_no_str + '_c2_test_scatterplot.png',  # Scatter Plot of Forecast vs Observation
        'c3' : path_result_plot + experiment_no_str + '_c3_test_residual_timeplot.png',  # Time Plot of Residual
        'c4' : path_result_plot + experiment_no_str + '_c4_test_residual_histogram.png',  # Histogram of Residual
        'c5' : path_result_plot + experiment_no_str + '_c5_test_learningcurve.png',  # Learning Curve vs Epoch
        
        # B. FOLDER FOR CROSS VALIDATION TIME SERIES
        'train_cv' : {
            1 : path_result_train + experiment_no_str + '_cv1_train_result.csv',
            2 : path_result_train + experiment_no_str + '_cv2_train_result.csv',
            3 : path_result_train + experiment_no_str + '_cv3_train_result.csv',
            4 : path_result_train + experiment_no_str + '_cv4_train_result.csv',
            5 : path_result_train + experiment_no_str + '_cv5_train_result.csv',
            6 : path_result_train + experiment_no_str + '_cv6_train_result.csv',
            7 : path_result_train + experiment_no_str + '_cv7_train_result.csv',
            8 : path_result_train + experiment_no_str + '_cv8_train_result.csv',
            9 : path_result_train + experiment_no_str + '_cv9_train_result.csv',
            10 : path_result_train + experiment_no_str + '_cv10_train_result.csv'
        },

        'test_cv' : {
            1 : path_result_test + experiment_no_str + '_cv1_test_result.csv',
            2 : path_result_test + experiment_no_str + '_cv2_test_result.csv',
            3 : path_result_test + experiment_no_str + '_cv3_test_result.csv',
            4 : path_result_test + experiment_no_str + '_cv4_test_result.csv',
            5 : path_result_test + experiment_no_str + '_cv5_test_result.csv',
            6 : path_result_test + experiment_no_str + '_cv6_test_result.csv',
            7 : path_result_test + experiment_no_str + '_cv7_test_result.csv',
            8 : path_result_test + experiment_no_str + '_cv8_test_result.csv',
            9 : path_result_test + experiment_no_str + '_cv9_test_result.csv',
            10 : path_result_test + experiment_no_str + '_cv10_test_result.csv'
        },
        
        'model' : {
            1 : path_model + experiment_no_str + '_cv1_model.pkl',
            2 : path_model + experiment_no_str + '_cv2_model.pkl',
            3 : path_model + experiment_no_str + '_cv3_model.pkl',
            4 : path_model + experiment_no_str + '_cv4_model.pkl',
            5 : path_model + experiment_no_str + '_cv5_model.pkl',
            6 : path_model + experiment_no_str + '_cv6_model.pkl',
            7 : path_model + experiment_no_str + '_cv7_model.pkl',
            8 : path_model + experiment_no_str + '_cv8_model.pkl',
            9 : path_model + experiment_no_str + '_cv9_model.pkl',
            10 : path_model + experiment_no_str + '_cv10_model.pkl'
        }
    }
    return hyperparameter,experiment_no_str, filepath

def export_result(filepath, df_a1_result, cross_val_result_df, hyperparameter):
    """Export experiment summary:
    1. experiment result
    2. hyperparameter
    3. cross validation detailed result

    Args:
        filepath (dict): dictionary of filepaths for exporting result
    """
    # Create a df of hyperparameter being used
    df_a2 = pd.DataFrame(hyperparameter)
    
    # EXPORT IT
    df_a1_result.to_csv(filepath['a1'], index=False)
    df_a2.to_csv(filepath['a2'])
    cross_val_result_df.to_csv(filepath['a3'])


# # DATA INPUT, CALENDAR FEATURE MAKING

# In[ ]:


# ADD NET LOAD HISTORICAL DATA
def add_lag_features(df, forecast_horizon, max_lag_day):
    """
    Adds a lagged column to the dataframe based on the given horizon in minutes and max lag in days.
    
    Args:
    df (pd.DataFrame): The input dataframe with a datetime index and a column 'y'.
    forecast_horizon (int): The horizon in minutes for the lag.
    max_lag_day (int): the number of days until the longest lag
    
    Returns:
    pd.DataFrame: The dataframe with additional columns for the lags.
    """
    
    # Convert the horizon to a timedelta object
    horizon_timedelta = pd.Timedelta(minutes=forecast_horizon)
    consecutive_timedelta = df.index[1] - df.index[0]
    
    # Calculate the number of new columns
    n_new_cols = len(df[df.index < df.index[0] + pd.DateOffset(days=max_lag_day)])
    
    # List to hold all the new lagged columns
    new_cols = []
    
    # Generate lagged columns based on the horizon and max lag
    
    #Generate lagged columns not only based on net load but also based on weather data if available
    for column in df.columns:
    # Generate lagged columns for the current column
        for i in range(n_new_cols):
            shift_timedelta = horizon_timedelta + i * consecutive_timedelta
            new_col_name = f'{column}_lag_{shift_timedelta}m'
            new_cols.append(df[column].shift(freq=shift_timedelta).rename(new_col_name))

    
    # Concatenate the new lagged columns with the original dataframe
    df = pd.concat([df] + new_cols, axis=1)
    
    df.dropna(inplace=True)
    
    return df


# In[ ]:


def separate_holdout(df, n_block):    
    """Separating df into two parts:
    1. df : df that will be used for training and blocked k-fold cross validation. 
    The block is a multiple of a week because net load data has weekly seasonality
    2. hold_out_df : this section is not used for now, but can be useful for final test of the chosen model
    if wanted, to show the generalized error. This is at least 1 block of data. 
    
    By default, the chosen k for k-fold cross validation is 10.
    
    For example, the original df has 12 weeks worth of data. 
    In this case,
    new df is week 1-10,
    hold_out_df is week 11-12,
    
    the new df will be used for cross validation, for example
    CV1: training: week 1-9, validation (test) week 10
    CV2: training: week 1-8, week 10, validation (test) week 9,
    etc. 

    Args:
        df (df): cleaned df consisting of y and all predictors
        n_block (int): number of blocks to divide the original df. This includes the block for hold_out_df, so if k=10, this n_block = k+1 = 11

    Returns:
        block_length (int) : number of weeks per block
        hodout_df (df) : unused df, can be used later for unbiased estimate of final model performance
        df (df) : df that will be used for training and validation (test) set
    """
    
    dataset_length_week= ((df.index[-1] - df.index[0]).total_seconds() / 86400/7)
    block_length = int(dataset_length_week / n_block)
    consecutive_timedelta = df.index[1] - df.index[0]
    n_timestep_per_week = int(one_week / consecutive_timedelta)
    holdout_start = (n_block - 1)* block_length * n_timestep_per_week
    holdout_df = df.iloc[holdout_start:]
    df = df.drop(df.index[holdout_start:])
    
    return block_length, holdout_df, df


# In[ ]:


def input_and_process(path_data_cleaned, forecast_horizon, max_lag_day, n_block, hyperparameter):
    """read dataset, add calendar features, add lag features (which depends on the forecast horizon).

    Args:
        path_data_cleaned (str): path to the dataset chosen
        forecast_horizon (int): forecast horizon in minutes
        max_lag_day (int): how much lag data will be used, written in days. For example, 7 means lag data until d-7 is used. 
        n_block (int): number of blocks to divide the original df. This includes the block for hold_out_df, so if k=10, this n_block = k+1 = 11
        hyperparameter (dict): hyperparameters for the model

    Returns:
        block_length (int): number of weeks per block
        holdout_df (df): unused df, can be used later for unbiased estimate of final model performance
        df (df): df that will be used for training and validation (test) set
    """
    # MAKE THIS AS FUNCTION
    # ADD CALENDAR DATA (holiday to add)
    # columns_to_use = ['datetime', 'netload_kW']
    df = pd.read_csv(path_data_cleaned + dataset, index_col=0, parse_dates=True)
    df.rename(columns={'netload_kW': 'y'}, inplace=True)
    
    # 1. Check if forecast horizon is >= dataset frequency
    # for example, if dataset is daily, forecast horizon should be at least 1 day
    # compute dataset frequency in minutes based on the datetime index
    dataset_freq = (df.index[1] - df.index[0]).seconds / 60
    if forecast_horizon < dataset_freq:
        raise ValueError('Forecast horizon should be >= dataset frequency')
    else:
        print('Pass Test 1 - Forecast horizon is >= dataset frequency')
        
    # 2. Check if hyperparameter choice is possible given the forecast horizon
    # for example, with forecast horizon of 2 days, we cannot use 1 day as the hyperparameter of seasonal naive forecast.

    
    if model_name == 'm2_snaive':
        if int(hyperparameter['days'] * 24 * 60) < forecast_horizon:
            raise ValueError('Choice of seasonal naive hyperparameter needs to be >= forecast horizon! Please change the hyperparameter.')
    # if model_name == 'm4_sarima':
    #     if int(hyperparameter['seasonal_period_days'] * 24 * 60) < forecast_horizon:
    #         raise ValueError('Choice of seasonal_period_days in SARIMA hyperparameter >= forecast horizon! Please change the hyperparameter.')
    print('Pass Test 2 - Hyperparameter choice is possible given the forecast horizon')
            
    
# ADD LAG FEATURES
    df = add_lag_features(df, forecast_horizon, max_lag_day)

# ADD CALENDAR FEATURES    
    # 1. Numerical representation of the datetime (Excel-style)
    numeric_datetime = pd.Series((df.index - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1), index=df.index)

    # 2. Year
    year = pd.Series(df.index.year, index=df.index)

    # 3. One-hot encoding of month (is_jan, is_feb, ..., is_nov, excluding December)
    month_dummies = pd.get_dummies(df.index.month, prefix='is', drop_first=False)

    # Custom column names for months: is_jan, is_feb, ..., is_nov
    month_names = ['is_jan', 'is_feb', 'is_mar', 'is_apr', 'is_may', 'is_jun', 
                'is_jul', 'is_aug', 'is_sep', 'is_oct', 'is_nov', 'is_dec']  

    # Drop the last column (December) to avoid redundancy and rename the columns
    month_dummies = month_dummies.iloc[:, :-1]  # Exclude December column
    month_dummies.columns = month_names[:month_dummies.shape[1]]  # Apply custom column names
    month_dummies = month_dummies.astype(int)  # Convert to 1 and 0
    month_dummies.index = df.index

    # 4. One-hot encoding of hour (hour_0, hour_1, ..., hour_22, excluding hour_23)
    hour_dummies = pd.get_dummies(df.index.hour, prefix='hour', drop_first=False).iloc[:, :-1]
    hour_dummies = hour_dummies.astype(int)  # Convert to 1 and 0
    hour_dummies.index = df.index

    # 5. One-hot encoding of day of week (is_mon, is_tue, ..., is_sat, excluding Sunday)
    # Mapping day of week (0=Mon, 1=Tue, ..., 6=Sun)
    dayofweek_dummies = pd.get_dummies(df.index.dayofweek, prefix='is', drop_first=False).iloc[:, :-1]

    # Custom mapping for days of the week: is_mon, is_tue, ..., is_sat
    dayofweek_names = ['is_mon', 'is_tue', 'is_wed', 'is_thu', 'is_fri', 'is_sat']  # Custom day names
    dayofweek_dummies.columns = dayofweek_names[:dayofweek_dummies.shape[1]]  # Apply custom column names
    dayofweek_dummies = dayofweek_dummies.astype(int)  # Convert to 1 and 0
    dayofweek_dummies.index = df.index

    # 6. Is weekday (1 if Monday to Friday, 0 if Saturday/Sunday)
    is_weekday = pd.Series((df.index.dayofweek < 5).astype(int), index=df.index)


    # Concatenate all new features into the original dataframe at once
    df = pd.concat([df, 
                    numeric_datetime.rename('numeric_datetime'), 
                    year.rename('year'),
                    month_dummies, 
                    hour_dummies, 
                    dayofweek_dummies, 
                    is_weekday.rename('is_weekday')], axis=1)
    
    block_length, holdout_df, df = separate_holdout(df, n_block)
    
    return block_length, holdout_df, df


# # CROSS VALIDATION

# In[ ]:


# SPLIT TRAIN - DEV - TEST SET
def split_time_series(df, cv_no):
    """Split df to train and test set using blocked cross validation.

    Args:
        df (df): df that will be used for training and validation (test) set, consists of X and Y
        cv_no (int): number of current cv order. 
                     cv_no=1 means the test set is at the last, cv_no = k means the test set is at the beginning

    Returns:
        train_df (df) : df used for training
        test_df (df) : df used for validation, formal name is validation set / dev set. 
    """
      
    n = len(df)
    test_start = int(n*(1 - cv_no*test_pct))
    test_end = int(n*(1 - (cv_no-1)*test_pct))
    
    test_df = df.iloc[test_start:test_end]
    train_df = df.drop(df.index[test_start:test_end])
    
    return train_df, test_df


# In[1]:


# SPLIT X AND y


# In[2]:


def split_xy(df):
    """separate forecast target y and all predictors X into two dfs

    Args:
        df (df): df containing the forecast target y and all predictors X
        
    Return:
        df_X (df): df of all predictors X
        df_y (df): df of target forecast y
    """

    df_y = df[['y']]
    df_X = df.drop("y", axis=1)
    
    return df_X, df_y


# # RUN MODEL

# In[ ]:


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


# In[ ]:


def train_model(model_name, hyperparameter, train_df_X, train_df_y, forecast_horizon):
    """train model based on the model choice, hyperparamter, predictors X and target y

    Args:
        model_name (string): eg 'm06_lr'
        hyperparameter (pd series): list of hyperparameter for that model
        train_df_X (df): matrix of predictors
        train_df_y (df): target forecast y
        forecast_horizon (int): Forecast horizon in minutes.

    Returns:
        model (dict) : general object storing all models info including the predictor, feature selection, 
        and all other relevant features
    """
    
    if model_name == 'm1_naive':
        model = train_model_m1_naive(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm2_snaive':
        model = train_model_m2_snaive(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm3_ets':
        model = train_model_m3_ets(hyperparameter, train_df_X, train_df_y, forecast_horizon)
    elif model_name == 'm4_arima':
        model = train_model_m4_arima(hyperparameter, train_df_X, train_df_y, forecast_horizon)
    elif model_name == 'm5_sarima':
        model = train_model_m5_sarima(hyperparameter, train_df_X, train_df_y, forecast_horizon)
    elif model_name == 'm6_lr':
        model = train_model_m6_lr(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm7_ann':
        model = train_model_m7_ann(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm8_dnn':
        model = train_model_m8_dnn(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm9_rt':
        model = train_model_m9_rt(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm10_rf':
        model = train_model_m10_rf(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm11_svr':
        model = train_model_m11_svr(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm12_rnn':
        model = train_model_m12_rnn(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm13_lstm':
        model = train_model_m13_lstm(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm14_gru':
        model = train_model_m14_gru(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm15_transformer':
        model = train_model_m15_transformer(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm16_prophet':
        model = train_model_m16_prophet(hyperparameter, train_df_X, train_df_y, forecast_horizon)
    elif model_name == 'm17_xgb':
        model = train_model_m17_xgb(hyperparameter, train_df_X, train_df_y)
    elif model_name == 'm18_nbeats':
        model = train_model_m18_nbeats(hyperparameter, train_df_X, train_df_y)
    else:
        raise ValueError(
            "Wrong Model Choice! Available models are: m1_naive, m2_snaive, m3_ets, m4_arima, m5_sarima, m6_lr, m7_ann, m8_dnn, m9_rt, m10_rf, m11_svr, m12_rnn, m13_lstm, m14_gru, m15_transformer, m16_prophet, m17_xgb, m18_nbeats"
        )

    return model


# In[ ]:


def produce_forecast(model_name, model, train_df_X, test_df_X, train_df_y, forecast_horizon):
    """Generate forecasts based on the model and its name.

    Args:
        model_name (string): Model identifier (e.g., 'm1_naive').
        model (dict): Trained model object containing all relevant features.
        train_df_X (DataFrame): Matrix of predictors for training set.
        test_df_X (DataFrame): Matrix of predictors for test set.
        train_df_y (DataFrame): Target forecast y for training set.
        forecast_horizon (int): Forecast horizon in minutes.

    Returns:
        train_df_y_hat (DataFrame): Forecasted values for training set.
        test_df_y_hat (DataFrame): Forecasted values for test set.
    """
    
    if model_name == 'm1_naive':
        train_df_y_hat, test_df_y_hat = produce_forecast_m1_naive(model, train_df_X, test_df_X)
    elif model_name == 'm2_snaive':
        train_df_y_hat, test_df_y_hat = produce_forecast_m2_snaive(model, train_df_X, test_df_X)
    elif model_name == 'm3_ets':
        train_df_y_hat, test_df_y_hat = produce_forecast_m3_ets(model, train_df_X, test_df_X, forecast_horizon)
    elif model_name == 'm4_arima':
        train_df_y_hat, test_df_y_hat = produce_forecast_m4_arima(model, train_df_X, test_df_X, forecast_horizon)
    elif model_name == 'm5_sarima':
        train_df_y_hat, test_df_y_hat = produce_forecast_m5_sarima(model, train_df_X, test_df_X, forecast_horizon)
    elif model_name == 'm6_lr':
        train_df_y_hat, test_df_y_hat = produce_forecast_m6_lr(model, train_df_X, test_df_X)
    elif model_name == 'm7_ann':
        train_df_y_hat, test_df_y_hat = produce_forecast_m7_ann(model, train_df_X, test_df_X)
    elif model_name == 'm8_dnn':
        train_df_y_hat, test_df_y_hat = produce_forecast_m8_dnn(model, train_df_X, test_df_X)
    elif model_name == 'm9_rt':
        train_df_y_hat, test_df_y_hat = produce_forecast_m9_rt(model, train_df_X, test_df_X)
    elif model_name == 'm10_rf':
        train_df_y_hat, test_df_y_hat = produce_forecast_m10_rf(model, train_df_X, test_df_X)
    elif model_name == 'm11_svr':
        train_df_y_hat, test_df_y_hat = produce_forecast_m11_svr(model, train_df_X, test_df_X)
    elif model_name == 'm12_rnn':
        train_df_y_hat, test_df_y_hat = produce_forecast_m12_rnn(model, train_df_X, test_df_X)
    elif model_name == 'm13_lstm':
        train_df_y_hat, test_df_y_hat = produce_forecast_m13_lstm(model, train_df_X, test_df_X)
    elif model_name == 'm14_gru':
        train_df_y_hat, test_df_y_hat = produce_forecast_m14_gru(model, train_df_X, test_df_X)
    elif model_name == 'm15_transformer':
        train_df_y_hat, test_df_y_hat = produce_forecast_m15_transformer(model, train_df_X, test_df_X)
    elif model_name == 'm16_prophet':
        train_df_y_hat, test_df_y_hat = produce_forecast_m16_prophet(model, train_df_X, test_df_X, train_df_y, forecast_horizon)
    elif model_name == 'm17_xgb':
        train_df_y_hat, test_df_y_hat = produce_forecast_m17_xgb(model, train_df_X, test_df_X)
    elif model_name == 'm18_nbeats':
        train_df_y_hat, test_df_y_hat = produce_forecast_m18_nbeats(model, train_df_X, test_df_X)
    else:
        raise ValueError(
            "Wrong Model Choice! Available models are: m1_naive, m2_snaive, m3_ets, m4_arima, m5_sarima, m6_lr, m7_ann, m8_dnn, m9_rt, m10_rf, m11_svr, m12_rnn, m13_lstm, m14_gru, m15_transformer, m16_prophet, m17_xgb, m18_nbeats"
        )

    return train_df_y_hat, test_df_y_hat


# In[ ]:


def save_model(filepath, cv_no, model):
    """Export model into binary file using pickle to a designated file

    Args:
        filepath (dictionary): dictionary of the file path
        cv_no (int) : cv number
        model (dictionary): trained model
    """
    
    with open(filepath['model'][cv_no], "wb") as model_file:
        # pickle.dump(model, model_file)
        dill.dump(model, model_file)


# In[ ]:


def run_model(df, model_name, hyperparameter, filepath, forecast_horizon, experiment_no_str, block_length):
    """Run model! This will be updated so that it can adapt to any model 
    This consists of
    1. Loop over all CV, then inside the loop,
    2. Split df to train and test set
    3. Train model on train set
    4. Produce naive forecast on both trian and test set for benchmark
    5. Produce forecast on both train and test set
    6. Compute residual on both train and test set
    7. Export the observation, naive, forecast, and residual of train and test set
    8. Produce plots only on CV 1
    9. Evaluate forecast performance on train and test set
    10. Quit the loop,
    11. Summarise the overall performance of the model using RMSE and its Stddev
    12. Export all results     
    
    Args:
        df (df): df that will be used for training and validation (test) set, consists of X and Y
        model_name (string): eg 'm06_lr'
        hyperparameter (pd series): list of hyperparameter for that model
        filepath (dict): dictionary of filepaths for exporting result
        forecast_horizon : the forecast horizon
        experiment_no : the experiment no. in string
        block_length : block length of one cross validation set
        
    
    """
    
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    cross_val_result_df = pd.DataFrame()

    # Compute max_y for normalization later
    max_y = df['y'].max()
    
    # DO CROSS VALIDATION
    for cv_no in range(1, k+1):
        print(f'Processing CV {cv_no} / {k}....')
        
        # SPLIT INTO TRAIN AND TEST X AND Y
        train_df, test_df = split_time_series(df, cv_no)
        train_df_X, train_df_y = split_xy(train_df)
        test_df_X, test_df_y = split_xy(test_df)

        # INITIALISE RESULT DF   
        train_result = train_df_y.copy()
        train_result = train_result.rename(columns={'y': 'observation'})

        test_result = test_df_y.copy()
        test_result = test_result.rename(columns={'y': 'observation'})

        # PRODUCE NAIVE FORECAST
        horizon_timedelta = pd.Timedelta(minutes=forecast_horizon)
        last_observation = f'y_lag_{horizon_timedelta}m'
        train_result['naive'] = train_df[last_observation]
        test_result['naive'] = test_df[last_observation]

        # TRAIN MODEL
        start_time = time.time()
        model = train_model(model_name, hyperparameter, train_df_X, train_df_y, forecast_horizon)
        save_model(filepath, cv_no, model)
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000  # Convert to milliseconds

        # PRODUCE FORECAST
        train_df_y_hat, test_df_y_hat = produce_forecast(model_name, model, train_df_X, test_df_X, train_df_y, forecast_horizon)
        train_result['forecast'] = train_df_y_hat
        test_result['forecast'] = test_df_y_hat
        
        # EVALUATE FORECAST
        train_result['residual'] = train_result['forecast'] - train_result['observation']
        test_result['residual'] = test_result['forecast'] - test_result['observation']
        train_R2 = compute_R2(train_result['forecast'], train_result['observation'])
        test_R2 = compute_R2(test_result['forecast'], test_result['observation'])
        
        train_RMSE = compute_RMSE(train_result['forecast'], train_result['observation'])
        test_RMSE = compute_RMSE(test_result['forecast'], test_result['observation'])
        
        train_nRMSE = 100*train_RMSE / max_y # in percent
        test_nRMSE = 100*test_RMSE / max_y # in percent
        
        cross_val_result = pd.DataFrame(
        {
            "runtime_ms": runtime_ms,
            "train_MBE": compute_MBE(train_result['forecast'], train_result['observation']), 
            "train_MAE": compute_MAE(train_result['forecast'], train_result['observation']),
            "train_RMSE": train_RMSE,
            "train_MAPE": compute_MAPE(train_result['forecast'], train_result['observation']),
            "train_MASE": compute_MASE(train_result['forecast'], train_result['observation'], train_result),
            "train_fskill": compute_fskill(train_result['forecast'], train_result['observation'], train_result['naive']),
            "train_R2": train_R2,
            "test_MBE": compute_MBE(test_result['forecast'], test_result['observation']),
            "test_MAE": compute_MAE(test_result['forecast'], test_result['observation']),
            "test_RMSE": test_RMSE,
            "test_MAPE": compute_MAPE(test_result['forecast'], test_result['observation']),
            "test_MASE": compute_MASE(test_result['forecast'], test_result['observation'], train_result),
            "test_fskill": compute_fskill(test_result['forecast'], test_result['observation'], test_result['naive']),
            "test_R2": test_R2,
            "train_nRMSE": train_nRMSE,
            "test_nRMSE": test_nRMSE
        }, 
        index=[cv_no]
        )
        
        if cross_val_result_df.empty:
            cross_val_result_df = cross_val_result
        else:
            cross_val_result_df = pd.concat([cross_val_result_df, cross_val_result], ignore_index=False)
        cross_val_result_df.index.name = 'cv_no'
        
        # EXPORT RESULTS DF TO CSV
        train_result.to_csv(filepath['train_cv'][cv_no])
        test_result.to_csv(filepath['test_cv'][cv_no])
        
        # IF CV_NO = 1, ALSO EXPORT SOME PLOTS
        if cv_no == 1:
            timeplot_forecast(train_result['observation'], train_result['forecast'], filepath['b1'])
            timeplot_forecast(test_result['observation'], test_result['forecast'], filepath['c1'])
            scatterplot_forecast(train_result['observation'], train_result['forecast'], train_R2, filepath['b2'])
            scatterplot_forecast(test_result['observation'], test_result['forecast'], test_R2, filepath['c2'])
            timeplot_residual(train_result['residual'], filepath['b3'])
            timeplot_residual(test_result['residual'], filepath['c3'])
            histogram_residual(train_result['residual'], df, filepath['b4'])
            histogram_residual(test_result['residual'], df, filepath['c4'])
        print()
        
        
    cross_val_result = pd.DataFrame(
        {
            "runtime_ms": [cross_val_result_df['runtime_ms'].mean(), cross_val_result_df['runtime_ms'].std()],
            "train_MBE": [cross_val_result_df['train_MBE'].mean(), cross_val_result_df['train_MBE'].std()], 
            "train_MAE": [cross_val_result_df['train_MAE'].mean(), cross_val_result_df['train_MAE'].std()],
            "train_RMSE": [cross_val_result_df['train_RMSE'].mean(), cross_val_result_df['train_RMSE'].std()],
            "train_MAPE": [cross_val_result_df['train_MAPE'].mean(), cross_val_result_df['train_MAPE'].std()],
            "train_MASE": [cross_val_result_df['train_MASE'].mean(), cross_val_result_df['train_MASE'].std()],
            "train_fskill": [cross_val_result_df['train_fskill'].mean(), cross_val_result_df['train_fskill'].std()],
            "train_R2": [cross_val_result_df['train_R2'].mean(), cross_val_result_df['train_R2'].std()],
            "test_MBE": [cross_val_result_df['test_MBE'].mean(), cross_val_result_df['test_MBE'].std()],
            "test_MAE": [cross_val_result_df['test_MAE'].mean(), cross_val_result_df['test_MAE'].std()],
            "test_RMSE": [cross_val_result_df['test_RMSE'].mean(), cross_val_result_df['test_RMSE'].std()],
            "test_MAPE": [cross_val_result_df['test_MAPE'].mean(), cross_val_result_df['test_MAPE'].std()],
            "test_MASE": [cross_val_result_df['test_MASE'].mean(), cross_val_result_df['test_MASE'].std()],
            "test_fskill": [cross_val_result_df['test_fskill'].mean(), cross_val_result_df['test_fskill'].std()],
            "test_R2": [cross_val_result_df['test_R2'].mean(), cross_val_result_df['test_R2'].std()],
            "train_nRMSE": [cross_val_result_df['train_nRMSE'].mean(), cross_val_result_df['train_nRMSE'].std()],
            "test_nRMSE": [cross_val_result_df['test_nRMSE'].mean(), cross_val_result_df['test_nRMSE'].std()]
        }, 
        index=['mean', 'stddev']
        )

    cross_val_result_df = pd.concat([cross_val_result_df, cross_val_result], ignore_index=False)

    data_a1 = {
        "experiment_no": experiment_no_str,
        "exp_date": datetime.today().strftime('%Y-%m-%d'), #today date in YYYY-MM-DD format
        "dataset_no": dataset.split('_')[0],
        "dataset": dataset.split('_')[1].split('.')[0],
        "dataset_freq_min": int((df.index[1] - df.index[0]).total_seconds() / 60),
        "dataset_length_week": block_length * (n_block - 1),
        "forecast_horizon_min": forecast_horizon,
        "train_pct": train_pct,
        "test_pct": test_pct,
        "model_no": model_name.split('_')[0],
        "hyperparameter_no": hyperparameter_no,
        "model_name": model_name + '_' + hyperparameter_no,
        "hyperparamter": ', '.join(f"{k}: {v}" for k, v in hyperparameter.items()),  
        "runtime_ms": cross_val_result_df.loc['mean', 'runtime_ms'],
        "train_RMSE": cross_val_result_df.loc['mean', 'train_RMSE'],
        "train_RMSE_stddev": cross_val_result_df.loc['stddev', 'train_RMSE'],
        "test_RMSE": cross_val_result_df.loc['mean', 'test_RMSE'],
        "test_RMSE_stddev": cross_val_result_df.loc['stddev', 'test_RMSE'],
        "train_nRMSE": cross_val_result_df.loc['mean', 'train_nRMSE'],
        "train_nRMSE_stddev": cross_val_result_df.loc['stddev', 'train_nRMSE'],
        "test_nRMSE": cross_val_result_df.loc['mean', 'test_nRMSE'],
        "test_nRMSE_stddev": cross_val_result_df.loc['stddev', 'test_nRMSE']
    }

    # Create a df of experiment result
    df_a1_result = pd.DataFrame([data_a1])
    
    export_result(filepath, df_a1_result, cross_val_result_df, hyperparameter)
    
    # return df_a1_result, cross_val_result_df


# In[ ]:


# RUN THE TOOL
def run_experiment(dataset, forecast_horizon, model_name, hyperparameter_no):
    '''
    Run the experiment with the specified parameters.
    Args:
        dataset (str): Name of the dataset file.
        forecast_horizon (int): Forecast horizon in minutes.
        model_name (str): Model identifier (e.g., 'm6_lr').
        hyperparameter_no (str): Hyperparameter set identifier.

    Returns:
        None. Results and models are saved to disk.
    '''
    # PREPARE FOLDER
    hyperparameter, experiment_no_str, filepath = prepare_directory(path_result, forecast_horizon, model_name, hyperparameter_no)
    # INPUT DATA
    block_length, holdout_df, df = input_and_process(path_data_cleaned, forecast_horizon, max_lag_day, n_block, hyperparameter)
    # RUN MODEL
    run_model(df, model_name, hyperparameter, filepath, forecast_horizon, experiment_no_str, block_length)


# # PERFORMANCE COMPUTATION

# In[3]:


# Mean Bias Error (MBE)
def compute_MBE(forecast, observation):
    """As the name suggest.

    Args:
        forecast (df): series of the forecast result from the model
        observation (df): series of the observed value (actual value)

    Returns:
        error as the name suggest (float): as the name suggest
    """
    return round(((forecast - observation).sum()) / len(observation), 5)

# Mean Absolute Error (MAE)
def compute_MAE(forecast, observation):
    """As the name suggest.

    Args:
        forecast (df): series of the forecast result from the model
        observation (df): series of the observed value (actual value)

    Returns:
        error as the name suggest (float): as the name suggest
    """
    return round((abs(forecast - observation)).mean(), 3)

# Root Mean Square Error (RMSE)
def compute_RMSE(forecast, observation):
    """As the name suggest.

    Args:
        forecast (df): series of the forecast result from the model
        observation (df): series of the observed value (actual value)

    Returns:
        error as the name suggest (float): as the name suggest
    """
    return round(np.sqrt(((forecast - observation) ** 2).mean()), 3)

# Mean Absolute Percentage Error (MAPE)
def compute_MAPE(forecast, observation):
    """As the name suggest. Be careful with MAPE though because its value can go to inf since the observed value can be 0. 

    Args:
        forecast (df): series of the forecast result from the model
        observation (df): series of the observed value (actual value)

    Returns:
        error as the name suggest (float): as the name suggest
    """
    return round((abs((forecast - observation) / observation) * 100).mean(), 3)

# Mean Absolute Scaled Error (MASE)
def compute_MASE(forecast, observation, train_result):
    """As the name suggest. MASE is first introduced by Rob Hyndman, used to handle MAPE problem being infinity. 
    Instead of using observed value as denominator,
    MASE uses MAE of the naive forecast at the train set for denominator. 

    Args:
        forecast (df): series of the forecast result from the model
        observation (df): series of the observed value (actual value)

    Returns:
        error as the name suggest (float): as the name suggest
    """
    errors = abs(forecast - observation)
    MAE_naive = compute_MAE(train_result['naive'], train_result['observation'])
    
    MASE = errors.mean() / MAE_naive
    return round(MASE, 3)

# Forecast Skill (FS)
def compute_fskill(forecast, observation, naive):
    """As the name suggest. Forecast Skill is a relative measure seeing the improvement 
    of the model performance over naive model. 

    Args:
        forecast (df): series of the forecast result from the model
        observation (df): series of the observed value (actual value)

    Returns:
        error as the name suggest (float): as the name suggest
    """
    return round((1 - compute_RMSE(forecast, observation) / compute_RMSE(naive, observation)) * 100, 3)

# R2
def compute_R2(forecast, observation):
    """As the name suggest. Be careful with R2 though because it is not a forecast evaluation. 
    It is just used to show linearity on the scatter plot of forecast and observed value. 

    Args:
        forecast (df): series of the forecast result from the model
        observation (df): series of the observed value (actual value)

    Returns:
        error as the name suggest (float): as the name suggest
    """
    return round(forecast.corr(observation)**2, 3)


# # PLOT

# In[ ]:


def timeplot_forecast(observation, forecast, pathname):
    """Produce time plot of observation vs forecast value and save it on the designated folder

    Args:
        observation (df): observed value
        forecast (df): forecast value
        pathname (str): filepath to save the figure
    """
    consecutive_timedelta = observation.index[-1] - observation.index[-2]
    # Calculate total minutes in a week
    minutes_per_week = 7 * 24 * 60  # 7 days * 24 hours * 60 minutes

    # Calculate the number of minutes per timestep
    minutes_per_timestep = consecutive_timedelta.total_seconds() / 60  # convert seconds to minutes

    # Compute the number of timesteps in a week
    timesteps_per_week = int(minutes_per_week / minutes_per_timestep)

    # Create the figure with specified size
    plt.figure(figsize=(9, 9))

    # Set background color
    # plt.gcf().patch.set_facecolor(platinum)

    # Plot the actual and forecast data
    plt.plot(observation[-timesteps_per_week:], color=dark_blue, label='Actual')
    plt.plot(forecast[-timesteps_per_week:], color=orange, label='Forecast')

    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'

    # Remove grid lines
    plt.grid(False)

    # Set tick marks for x and y axis
    plt.xticks(fontsize=12, color=dark_blue, alpha=0.5, rotation=30)
    plt.yticks(fontsize=12, color=dark_blue, alpha=0.5)

    # Add borders to the plot
    plt.gca().spines['top'].set_color(dark_blue)
    plt.gca().spines['right'].set_color(dark_blue)
    plt.gca().spines['bottom'].set_color(dark_blue)
    plt.gca().spines['left'].set_color(dark_blue)

    # Remove the tick markers (the small lines)
    plt.tick_params(axis='x', which='both', length=0)  # Remove x-axis tick markers
    plt.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick markers

    # Set axis titles
    plt.xlabel('Time', fontsize=14, color=dark_blue)
    plt.ylabel('Net Load (kW)', fontsize=14, color=dark_blue)

    # Remove title
    plt.title('')

    plt.legend(loc='upper left', fontsize=12, frameon=False, labelspacing=1, bbox_to_anchor=(1, 1))

    plt.savefig(pathname, format='png', bbox_inches='tight')
    plt.close()



    # Show the plot
    # plt.show()


# In[ ]:


def scatterplot_forecast(observation, forecast, R2, pathname):
    """Produce scatterplot observation vs forecast value and save it on the designated folder

    Args:
        observation (df): observed value
        forecast (df): forecast value
        pathname (str): filepath to save the figure
    """
    # Create the figure with specified size
    plt.figure(figsize=(9, 9))

    # Set background color
    # plt.gcf().patch.set_facecolor(platinum)

    # Plot the actual and forecast data
    plt.scatter(forecast, observation, color=dark_blue, label='Actual', s=40, alpha=0.7)  # 's' sets the size of the points


    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'

    # Remove grid lines
    plt.grid(False)

    # Set tick marks for x and y axis
    plt.xticks(fontsize=12, color=dark_blue, alpha=0.5, rotation=0)
    plt.yticks(fontsize=12, color=dark_blue, alpha=0.5)

    # Add borders to the plot
    plt.gca().spines['top'].set_color(dark_blue)
    plt.gca().spines['right'].set_color(dark_blue)
    plt.gca().spines['bottom'].set_color(dark_blue)
    plt.gca().spines['left'].set_color(dark_blue)

    # Remove the tick markers (the small lines)
    plt.tick_params(axis='x', which='both', length=0)  # Remove x-axis tick markers
    plt.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick markers

    # Set axis titles
    plt.xlabel('Net Load Forecast (kW)', fontsize=14, color=dark_blue)
    plt.ylabel('Net Load Observation (kW)', fontsize=14, color=dark_blue)

    # Remove title
    plt.title('')
    
    # Add R² value at the top-left corner
    plt.text(0.95, 0.05, f'R² = {R2:.3f}', transform=plt.gca().transAxes, 
         fontsize=14, color=dark_blue, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor=dark_blue, boxstyle='round,pad=0.5', linewidth=1))


    plt.savefig(pathname, format='png', bbox_inches='tight')
    plt.close()



    # Show the plot
#     plt.show()


# In[ ]:


def timeplot_residual(residual, pathname):
    """Produce time plot of resodia; value and save it on the designated folder

    Args:
        residual (df): forecast - observation
        pathname (str): filepath to save the figure
    """
    consecutive_timedelta = residual.index[-1] - residual.index[-2]
    # Calculate total minutes in a week
    minutes_per_week = 7 * 24 * 60  # 7 days * 24 hours * 60 minutes

    # Calculate the number of minutes per timestep
    minutes_per_timestep = consecutive_timedelta.total_seconds() / 60  # convert seconds to minutes

    # Compute the number of timesteps in a week
    timesteps_per_week = int(minutes_per_week / minutes_per_timestep)

    # Create the figure with specified size
    plt.figure(figsize=(9, 9))

    # Set background color
    # plt.gcf().patch.set_facecolor(platinum)

    # Plot the actual and forecast data
    plt.plot(residual[-timesteps_per_week:], color=dark_blue, label='Actual')

    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'

    # Remove grid lines
    plt.grid(False)

    # Set tick marks for x and y axis
    plt.xticks(fontsize=12, color=dark_blue, alpha=0.5, rotation=30)
    plt.yticks(fontsize=12, color=dark_blue, alpha=0.5)

    # Add borders to the plot
    plt.gca().spines['top'].set_color(dark_blue)
    plt.gca().spines['right'].set_color(dark_blue)
    plt.gca().spines['bottom'].set_color(dark_blue)
    plt.gca().spines['left'].set_color(dark_blue)

    # Remove the tick markers (the small lines)
    plt.tick_params(axis='x', which='both', length=0)  # Remove x-axis tick markers
    plt.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick markers

    # Set axis titles
    plt.xlabel('Time', fontsize=14, color=dark_blue)
    plt.ylabel('Forecast Residual (kW)', fontsize=14, color=dark_blue)

    # Remove title
    plt.title('')

    plt.savefig(pathname, format='png', bbox_inches='tight')
    plt.close()



    # Show the plot
    # plt.show()


# In[ ]:


def histogram_residual(residual, df, pathname):
    """Produce histogiram of residual value and save it on the designated folder

    Args:
        residual (df): forecast - observation
        pathname (str): filepath to save the figure
    """
    # Create the figure with specified size
    plt.figure(figsize=(9, 9))

    # Set background color
    # plt.gcf().patch.set_facecolor(platinum)

    # Compute the range
    dataset_range = df['y'].max() - df['y'].min()
    bin_min = -dataset_range/7
    bin_max = dataset_range/7
    
    # Plot the actual and forecast data
    plt.hist(residual, bins=31, range=(bin_min, bin_max), color=dark_blue, edgecolor=dark_blue, alpha=0.7)
    

    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'

    # Remove grid lines
    plt.grid(False)

    # Set tick marks for x and y axis
    plt.xticks(fontsize=12, color=dark_blue, alpha=0.5, rotation=0)
    plt.yticks(fontsize=12, color=dark_blue, alpha=0.5)

    # Add borders to the plot
    plt.gca().spines['top'].set_color(dark_blue)
    plt.gca().spines['right'].set_color(dark_blue)
    plt.gca().spines['bottom'].set_color(dark_blue)
    plt.gca().spines['left'].set_color(dark_blue)

    # Remove the tick markers (the small lines)
    plt.tick_params(axis='x', which='both', length=0)  # Remove x-axis tick markers
    plt.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick markers

    # Set axis titles
    plt.xlabel('Forecast Residual (kW)', fontsize=14, color=dark_blue)
    plt.ylabel('Count', fontsize=14, color=dark_blue)

    # Remove title
    plt.title('')


    plt.savefig(pathname, format='png', bbox_inches='tight')
    plt.close()



    # Show the plot
    # plt.show()

