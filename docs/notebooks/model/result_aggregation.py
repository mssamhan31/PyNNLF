#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from pathlib import Path

# Create an empty Df
df = pd.DataFrame()
exp_result_folder = '../../experiment_result/'

# Compute the number of folder in the experiment result folder 
def get_number_of_folders(path):
    return len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

# Get the number of folders in the experiment result folder
number_of_folders = get_number_of_folders(exp_result_folder)
n_experiments = number_of_folders - 1

parent = Path(exp_result_folder)        # adjust

for i in range(1, n_experiments + 1):        # 1 → n (change to 0-based if you need)
    # “E” + 5-digit zero-padded index, e.g. 1 → E00001, 12 → E00012, 123 → E00123
    pattern = f"E{i:05d}_*"                  # tail “_*” matches any suffix
    for folder in parent.glob(pattern):      # returns Path objects for every match
        print(folder)                        # ↪ or os.chdir(folder) / open files here
        try:
            # any file that ends with “…_experiment_result[.csv]”
            csv_path = next(folder.glob("*_experiment_result*.csv"))
            row_df = pd.read_csv(csv_path, nrows=1)     # one-row DataFrame
            df = pd.concat([df, row_df], ignore_index=True)
        except StopIteration:
            print(f"⚠️ no result file in {folder}")
        except Exception as e:
            print(f"⚠️ error reading {folder}: {e}")
            
# export df to excel file on experiment_result folder
output_file = Path(exp_result_folder) / 'result_summary.xlsx'
df.to_excel(output_file, sheet_name='result', index=False)

