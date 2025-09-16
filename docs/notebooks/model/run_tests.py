#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 1. RUN CONFIG: FUNCTIONS, CONSTANTS, MODELS, ETC
get_ipython().run_line_magic('run', '"../config/config.ipynb"')

list_dataset = [ds0]
list_forecast_horizon = [fh1] # 30min ahead

list_model_and_hp = [
    [m1, 'hp1'],
    [m2, 'hp2'],
    [m3, 'hp1'],
    [m4, 'hp1'],
    [m5, 'hp1'],
    [m6, 'hp1'],
    [m7, 'hp1'],
    [m8, 'hp1'],
    [m9, 'hp3'],
    [m10, 'hp1'],
    [m11, 'hp1'],
    [m12, 'hp1'],
    [m13, 'hp1'],
    [m14, 'hp1'],
    [m15, 'hp1'],
    [m16, 'hp1'],
    [m17, 'hp1'],
    [m18, 'hp1']
]


# In[4]:


# RUN BATCH FOR ALL COMBINATIONS
for dataset in list_dataset:
    for forecast_horizon in list_forecast_horizon:
        for model_name, hyperparameter_no in list_model_and_hp:
            print(f"Running {model_name} with hyperparameter {hyperparameter_no} on {dataset} for forecast horizon {forecast_horizon}")
            dataset = dataset
            forecast_horizon = forecast_horizon # fh1 = 30 minutes ahead, fh9 = 2 days ahead

            # MODEL AND HYPERPARAMETER TO CHOOSE
            model_name = model_name
            hyperparameter_no = hyperparameter_no
            
            # 3. RUN EXPERIMENT
            run_experiment(dataset, forecast_horizon, model_name, hyperparameter_no)


# # Populate Last 18 Experiment Cross Validation Results For Testing

# In[7]:


import pandas as pd
import os
from pathlib import Path
import numpy as np
from datetime import date

exp_result_folder = '../../experiment_result/'
benchmark_file = '../../experiment_result/Archive/Testing Result/testing_benchmark.xlsx'
output_folder = '../../experiment_result/Archive/Testing Result/'

# Get m1-m18 name

# Create an empty Df
df = pd.DataFrame()


# Compute the number of folder in the experiment result folder 
def get_number_of_folders(path):
    return len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

# Get the number of folders in the experiment result folder
number_of_folders = get_number_of_folders(exp_result_folder)
n_experiments = number_of_folders - 1
pos_first_naive = n_experiments - 18 + 1
parent = Path(exp_result_folder)        # adjust

summary_df = pd.DataFrame()

for i in range(pos_first_naive, n_experiments + 1):        # 1 → n (change to 0-based if you need)
    # “E” + 5-digit zero-padded index, e.g. 1 → E00001, 12 → E00012, 123 → E00123
    pattern = f"E{i:05d}_*"                  # tail “_*” matches any suffix
    for folder in parent.glob(pattern):      # returns Path objects for every match
        print('Extracting ' + str(folder))                        # ↪ or os.chdir(folder) / open files here
        
        # Extract model name
        
        import re

        match = re.search(r"(m\d+_[^_]+)", str(folder))
        if match:
            model = match.group(1)
        
        try:
            # any file that ends with “…_cross_validation_result[.csv]”
            csv_path = next(folder.glob("*_cross_validation_result*.csv"))
            result_df = pd.read_csv(csv_path, header=0, skiprows=range(1, 11), nrows=2, index_col=0).T     #  result
            
            result_df = result_df.stack()
            result_df.index = result_df.index.map(lambda x: f"{x[0]}_{x[1]}")
            result_df = result_df.to_frame(name="value, test")
            result_df.index = model + "_" + result_df.index.astype(str)

            summary_df = pd.concat([summary_df, result_df], axis=0)

        except StopIteration:
            print(f"⚠️ no result file in {folder}")
        except Exception as e:
            print(f"⚠️ error reading {folder}: {e}")

# Copy benchmark values (from xlsx, specific sheet & columns)
benchmark_df = pd.read_excel(
    benchmark_file,
    sheet_name='testing_benchmark',
    index_col=0,
    usecols=['Unnamed: 0', 'min acceptable value', 'max acceptable value']
)

# Merge with summary_df
summary_df = pd.concat([benchmark_df, summary_df], axis=1)

# Round all numeric columns to nearest 2 decimal places
summary_df[summary_df.select_dtypes(include=[np.number]).columns] = (
    summary_df.select_dtypes(include=[np.number]).round(2)
)

# Test results: pass if within [min, max], otherwise fail
summary_df['test_result'] = np.where(
    (summary_df['value, test'] >= summary_df['min acceptable value']) &
    (summary_df['value, test'] <= summary_df['max acceptable value']),
    'pass',
    'outside of the acceptable value'
)

# Create output file name
output_file = f"{date.today().strftime('%Y%m%d')}_test_result.csv"

# Output testing file result
print('Exporting ' + output_file)  # e.g. 20250819_test_result.csv
summary_df.to_csv(output_folder + output_file, index=True)

