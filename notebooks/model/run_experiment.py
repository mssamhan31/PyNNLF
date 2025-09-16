#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. RUN CONFIG
get_ipython().run_line_magic('run', '"../config/config.ipynb"')

# 2. SETUP FORECAST PROBLEM AND MODEL SPECIFICATION (USER TO INPUT)
# FORECAST PROBLEM
dataset = ds0
forecast_horizon = fh1 # fh1 = 30 minutes ahead, fh9 = 2 days ahead
# MODEL SPECIFICATION
model_name = m6
hyperparameter_no = 'hp1'

# 3. RUN EXPERIMENT
run_experiment(dataset, forecast_horizon, model_name, hyperparameter_no)


# In[ ]:


mkdocs --version


# In[5]:


get_ipython().run_line_magic('pip', 'install mkdocs')

