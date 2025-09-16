#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. RUN CONFIG: FUNCTIONS, CONSTANTS, MODELS, ETC
get_ipython().run_line_magic('run', '"../config/config.ipynb"')

list_dataset = [ds15]
list_forecast_horizon = [fh8] # 1 day ahead

list_model_and_hp = [
    [m1, 'hp1'],
    [m2, 'hp2'],
    [m3, 'hp1'],
    [m4, 'hp1'],
    [m6, 'hp1'],
    [m7, 'hp1'],
    [m8, 'hp1'],
    [m9, 'hp3'],
    [m10, 'hp1'],
    [m13, 'hp2'],
    [m16, 'hp1'],
    [m17, 'hp1']
]


# In[ ]:


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


# Note
# 
# To run the model properly, there must be no NAs on the dataset, and the timesteps must be complete and in regular interval.

# In[ ]:




