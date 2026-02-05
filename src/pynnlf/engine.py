import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import os
import time
import pickle # for saving trained model
import dill # for saving trained model
import importlib.util
from pathlib import Path   
import re
 


# # FOLDER PREPARATION



def load_model_module(models_dir: Path, model_name: str):
    """
    Load a model module from the workspace models directory.

    Args:
        models_dir (Path): <workspace>/models
        model_name (str): e.g., "m6_lr"

    Returns:
        module: Imported python module object.
    """
    p = models_dir / f"{model_name}.py"
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    spec = importlib.util.spec_from_file_location(model_name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def compute_exp_no(path_result):
    """Compute experiment number for folder & file naming.

    This version:
    1) Detects existing experiment folders matching pattern: E00001_*
    2) Uses the maximum existing number + 1
    3) Starts numbering from 1
    4) Ignores Archive/other folders/files safely

    Args:
        path_result (str): path to experiment_result folder

    Returns:
        int: experiment_no (starts at 1)
        str: experiment_no_str (e.g., "E00001")
    """
    if not os.path.exists(path_result):
        os.makedirs(path_result, exist_ok=True)

    pat = re.compile(r"^E(\d{5})_")
    nums = []

    for name in os.listdir(path_result):
        full = os.path.join(path_result, name)
        if not os.path.isdir(full):
            continue
        m = pat.match(name)
        if m:
            nums.append(int(m.group(1)))

    # Start from 1 if none exist
    next_no = (max(nums) + 1) if nums else 1
    next_no_str = f"E{next_no:05d}"
    return next_no, next_no_str

def compute_folder_name(experiment_no_str, dataset_file, forecast_horizon, model_name, hyperparameter_no):
    """
    Folder name in the format of [exp number]_[exp date]_[dataset]_[forecast horizon]_[model]_[hyperparameter]

    Args:
        experiment_no_str (str): exp number
        dataset_file (str): dataset filename (e.g., "ds0_test.csv")
        forecast_horizon (int): forecast horizon in minutes
        model_name (str): for example, m6_lr
        hyperparameter_no (str): for example, hp1

    Returns:
        str: folder name
    """
    folder_name = (
        experiment_no_str + '_' +
        datetime.today().date().strftime("%y%m%d") + '_' +
        dataset_file.split('_')[0] + '_' +
        'fh' + str(forecast_horizon) + '_' +
        model_name + '_' +
        hyperparameter_no
    )
    return folder_name

def prepare_directory(path_result, dataset_file, forecast_horizon, model_name, hyperparameter_no, hyperparameter_dict):
    """
    Create experiment folders and filepaths for exports.

    Args:
        path_result (str): path to experiment_result folder
        dataset_file (str): dataset filename
        forecast_horizon (int): forecast horizon in minutes
        model_name (str): model name e.g., m6_lr
        hyperparameter_no (str): e.g., hp1
        hyperparameter_dict (dict): chosen hyperparameter dict

    Returns:
        dict: hyperparameter_dict
        str: experiment_no_str
        dict: filepath dictionary
    """
    hyperparameter = hyperparameter_dict

    experiment_no, experiment_no_str = compute_exp_no(path_result)
    folder_name = compute_folder_name(experiment_no_str, dataset_file, forecast_horizon, model_name, hyperparameter_no)

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
    os.makedirs(path_result2, exist_ok=False)
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
    # Create a df of hyperparameter being used (dict -> key/value table)
    df_a2 = pd.DataFrame(
        {"hyperparameter": list(hyperparameter.keys()),
        "value": list(hyperparameter.values())}
    )
    
    # EXPORT IT
    df_a1_result.to_csv(filepath['a1'], index=False)
    df_a2.to_csv(filepath['a2'])
    cross_val_result_df.to_csv(filepath['a3'])


# # DATA INPUT, CALENDAR FEATURE MAKING

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
    
    one_week = dt.timedelta(weeks=1)
    dataset_length_week= ((df.index[-1] - df.index[0]).total_seconds() / 86400/7)
    block_length = int(dataset_length_week / n_block)
    consecutive_timedelta = df.index[1] - df.index[0]
    n_timestep_per_week = int(one_week / consecutive_timedelta)
    holdout_start = (n_block - 1)* block_length * n_timestep_per_week
    holdout_df = df.iloc[holdout_start:]
    df = df.drop(df.index[holdout_start:])
    
    return block_length, holdout_df, df

def input_and_process(dataset_path, model_name, forecast_horizon, max_lag_day, n_block, hyperparameter):
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
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
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
# SPLIT TRAIN - DEV - TEST SET
def split_time_series(df, cv_no, test_pct):
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




# SPLIT X AND y




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

def call_train(train_fn, hyperparameter, train_df_X, train_df_y, forecast_horizon):
    """
    Call a model's train function robustly across different signatures.

    Args:
        train_fn (callable): train_model_<model_name> function
        hyperparameter (dict): hyperparameter dict from YAML
        train_df_X (df): predictors
        train_df_y (df): target
        forecast_horizon (int): minutes

    Returns:
        object: model object returned by the model's train function
    """
    try:
        return train_fn(hyperparameter, train_df_X, train_df_y, forecast_horizon)
    except TypeError:
        return train_fn(hyperparameter, train_df_X, train_df_y)

def call_forecast(forecast_fn, model, train_df_X, test_df_X, train_df_y, forecast_horizon):
    """
    Call a model's forecast function robustly across different signatures.

    Args:
        forecast_fn (callable): produce_forecast_<model_name> function
        model (object): trained model object
        train_df_X (df): predictors for train
        test_df_X (df): predictors for test
        train_df_y (df): target for train (needed by Prophet)
        forecast_horizon (int): minutes

    Returns:
        tuple: (train_df_y_hat, test_df_y_hat)
    """
    # Most general first (Prophet-like)
    try:
        return forecast_fn(model, train_df_X, test_df_X, train_df_y, forecast_horizon)
    except TypeError:
        pass
    # Statsmodels-like
    try:
        return forecast_fn(model, train_df_X, test_df_X, forecast_horizon)
    except TypeError:
        pass
    # Simple ML models
    return forecast_fn(model, train_df_X, test_df_X)


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

import numpy as np
import pandas as pd

def to_series(y_hat, target_index):
    """
    Convert model output to a 1D pandas Series aligned to target_index.

    Rules:
    - If y_hat is Series/DataFrame with its own index: reindex to target_index.
    - If y_hat is array-like: require matching length (otherwise raise).

    Args:
        y_hat (any): model output (np array / Series / DataFrame)
        target_index (pd.Index): desired index

    Returns:
        pd.Series: forecast aligned to target_index
    """
    # DataFrame -> Series
    if isinstance(y_hat, pd.DataFrame):
        if y_hat.shape[1] == 1:
            s = y_hat.iloc[:, 0]
        else:
            # flatten multi-col to 1D (defensive)
            s = pd.Series(np.asarray(y_hat).ravel(), index=y_hat.index[:len(np.asarray(y_hat).ravel())])
    elif isinstance(y_hat, pd.Series):
        s = y_hat
    else:
        arr = np.asarray(y_hat).ravel()
        if len(arr) != len(target_index):
            raise ValueError(
                f"Forecast length mismatch: got {len(arr)} values, expected {len(target_index)}"
            )
        return pd.Series(arr, index=target_index)

    # If it has an index, align by timestamps
    if hasattr(s, "index") and len(s.index) > 0:
        return s.reindex(target_index)

    # Fallback (should rarely happen)
    arr = np.asarray(s).ravel()
    if len(arr) != len(target_index):
        raise ValueError(
            f"Forecast length mismatch: got {len(arr)} values, expected {len(target_index)}"
        )
    return pd.Series(arr, index=target_index)
        
def run_model(
    df,
    model_mod,
    model_name,
    hyperparameter,
    filepath,
    forecast_horizon,
    experiment_no_str,
    block_length,
    *,
    dataset_file,
    hyperparameter_no,
    k,
    test_pct,
    train_pct,
    n_block,
    plot_enabled,
    plot_style,
):
    """
    Run CV loop, train model, forecast, export outputs.

    Args:
        df (df): processed dataframe used for CV
        model_mod (module): loaded model module from workspace
        model_name (str): e.g. "m6_lr"
        hyperparameter (dict): hp dict
        filepath (dict): export paths from prepare_directory
        forecast_horizon (int): minutes
        experiment_no_str (str): e.g. "E00001"
        block_length (int): weeks per block

        dataset_file (str): dataset filename e.g. "ds0_test.csv"
        hyperparameter_no (str): e.g. "hp1"
        k (int): number of CV folds
        test_pct (float): 1/k
        train_pct (float): 1 - test_pct
        n_block (int): k + 1
        plot_enabled (bool): plot on/off
        plot_style (dict): colors + font

    Returns:
        None
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
        train_df, test_df = split_time_series(df, cv_no, test_pct)
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

        # CALL TRAIN AND FORECAST PRODUCTION
        train_fn = getattr(model_mod, f"train_model_{model_name}")
        forecast_fn = getattr(model_mod, f"produce_forecast_{model_name}")

        # TRAIN MODEL
        start_time = time.time()
        model = call_train(train_fn, hyperparameter, train_df_X, train_df_y, forecast_horizon)
        save_model(filepath, cv_no, model)
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000

        # PRODUCE FORECAST
        train_df_y_hat, test_df_y_hat = call_forecast(
            forecast_fn, model, train_df_X, test_df_X, train_df_y, forecast_horizon
        )
        
        # NORMALIZE FORECAST OUTPUTS
        train_result["forecast"] = to_series(train_df_y_hat, train_df_X.index)
        test_result["forecast"]  = to_series(test_df_y_hat,  test_df_X.index)
          
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
        dark_blue = plot_style["colors"]["dark_blue"]
        orange = plot_style["colors"]["orange"]
        plt.rcParams["font.family"] = plot_style["font_family"]
        
        # PLOTS (CV1 only)
        if plot_enabled and cv_no == 1:
            dark_blue = plot_style["colors"]["dark_blue"]
            orange = plot_style["colors"]["orange"]
            plt.rcParams["font.family"] = plot_style["font_family"]

            timeplot_forecast(train_result["observation"], train_result["forecast"], filepath["b1"], dark_blue, orange)
            timeplot_forecast(test_result["observation"],  test_result["forecast"],  filepath["c1"], dark_blue, orange)
            scatterplot_forecast(train_result["observation"], train_result["forecast"], train_R2, filepath["b2"], dark_blue, orange)
            scatterplot_forecast(test_result["observation"],  test_result["forecast"],  test_R2, filepath["c2"], dark_blue, orange)
            timeplot_residual(train_result["residual"], filepath["b3"], dark_blue, orange)
            timeplot_residual(test_result["residual"],  filepath["c3"], dark_blue, orange)
            histogram_residual(train_result["residual"], df, filepath["b4"], dark_blue, orange)
            histogram_residual(test_result["residual"],  df, filepath["c4"], dark_blue, orange)
        
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
        "dataset_no": dataset_file.split('_')[0],
        "dataset": dataset_file.split('_')[1].split('.')[0] if '_' in dataset_file else dataset_file.split('.')[0],
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

def validate_model_module(model_mod, model_name: str) -> None:
    """
    Validate that a model module provides required functions.

    Args:
        model_mod (module): loaded model module
        model_name (str): file stem, e.g. "m6_lr"

    Returns:
        None
    """
    train_name = f"train_model_{model_name}"
    fcst_name = f"produce_forecast_{model_name}"
    if not hasattr(model_mod, train_name):
        raise AttributeError(f"Missing function '{train_name}' in {model_name}.py")
    if not hasattr(model_mod, fcst_name):
        raise AttributeError(f"Missing function '{fcst_name}' in {model_name}.py")
    
# RUN THE TOOL
def run_experiment_engine(
    dataset_path,
    forecast_horizon_min,
    model_name,
    hyperparameter_no,
    hyperparameter,
    output_dir,
    models_dir,
    config,
):
    """
    Run one experiment end-to-end using explicit inputs (no notebook globals).

    Args:
        dataset_path (str | Path): full path to dataset CSV (workspace/data/...)
        forecast_horizon_min (int): forecast horizon in minutes
        model_name (str): model name e.g. "m6_lr"
        hyperparameter_no (str): hp identifier e.g. "hp1"
        hyperparameter (dict): hyperparameter dict for this run
        output_dir (str | Path): workspace/experiment_result
        models_dir (str | Path): workspace/models (workspace-only)
        config (dict): parsed workspace/specs/pynnlf_config.yaml (cv + plot + paths + registries)

    Returns:
        None
    """
    dataset_path = Path(dataset_path)
    dataset_file = dataset_path.name

    # CV config
    k = int(config["cv"]["k"])
    test_pct = 1 / k
    train_pct = 1 - test_pct
    n_block = k + 1
    max_lag_day = int(config["cv"]["max_lag_day"])

    # load workspace model module
    model_mod = load_model_module(Path(models_dir), model_name)
    validate_model_module(model_mod, model_name)

    # folders + filepaths
    hyperparameter_used, experiment_no_str, filepath = prepare_directory(
        str(Path(output_dir)) + "/",
        dataset_file,
        forecast_horizon_min,
        model_name,
        hyperparameter_no,
        hyperparameter,
    )

    # data prep
    block_length, holdout_df, df = input_and_process(
        dataset_path,
        model_name,
        forecast_horizon_min,
        max_lag_day,
        n_block,
        hyperparameter,
    )

    # run CV + export a1/a2/a3
    run_model(
        df,
        model_mod,
        model_name,
        hyperparameter,
        filepath,
        forecast_horizon_min,
        experiment_no_str,
        block_length,
        dataset_file=dataset_file,
        hyperparameter_no=hyperparameter_no,
        k=k,
        test_pct=test_pct,
        train_pct=train_pct,
        n_block=n_block,
        plot_enabled=bool(config["plot"]["enabled"]),
        plot_style=config["plot"],
    )

# # PERFORMANCE COMPUTATION



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
def timeplot_forecast(observation, forecast, pathname, dark_blue, orange):
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

def scatterplot_forecast(observation, forecast, R2, pathname, dark_blue, orange):
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

def timeplot_residual(residual, pathname, dark_blue, orange):
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

def histogram_residual(residual, df, pathname, dark_blue, orange):
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

