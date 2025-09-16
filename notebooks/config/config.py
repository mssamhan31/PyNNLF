#!/usr/bin/env python
# coding: utf-8

# ## Goal
# 1. Populate all hard coded path, file name, values, including hyperparameters

# # DATASET NAME

# In[ ]:


ds0 = 'ds0_test.csv'
ds1 = 'ds1_ashd.csv'
ds2 = 'ds2_aedp_5min.csv'
ds3 = 'ds3_aedp_30min.csv'
ds4 = 'ds4_ashd_with_weather.csv'
ds5 = 'ds5_aedp_30min_with_weather.csv'
ds6 = 'ds6_aedp_cluster_5min.csv'
ds7 = 'ds7_aedp_cluster_30min.csv'
ds8 = 'ds8_aedp_cluster_30min_with_weather.csv'
ds9 = 'ds9_aedp_cluster2_5min.csv'
ds10 = 'ds10_aedp_cluster2_30min.csv'
ds11 = 'ds11_aedp_cluster2_30min_with_weather.csv'
ds12 = 'ds12_ashd_with_cloud_bom.csv'
ds13 = 'ds13_ashd_with_cloud_solcast.csv'
ds14 = 'ds14_ausgrid_zs_mascot.csv'
ds15 = 'ds15_ausgrid_zs_mascot_30min_with_weather.csv'


# # FORECAST HORIZON

# In[ ]:


#all in minutes
fh1 = 30
fh2 = 60
fh3 = 120
fh4 = 180
fh5 = 240
fh6 = 300
fh7 = 360
fh8 = 1440 # 1 day
fh9 = 2880 # 2 days
fh10 = 10080 # 1 week
fh11 = 43200 # 1 month 


# In[ ]:


get_ipython().run_line_magic('run', '"../config/constant.ipynb"')
get_ipython().run_line_magic('run', '"../config/model_hyperparameters.ipynb"')
get_ipython().run_line_magic('run', '"../config/general_functions.ipynb"')


# In[ ]:


# MODELS
# Custom models
get_ipython().run_line_magic('run', '"../model/m1_naive.ipynb"')
get_ipython().run_line_magic('run', '"../model/m2_snaive.ipynb"')

# Statsmodels-based models
get_ipython().run_line_magic('run', '"../model/m3_ets.ipynb"')
get_ipython().run_line_magic('run', '"../model/m4_arima.ipynb"')
get_ipython().run_line_magic('run', '"../model/m5_sarima.ipynb"')

# Scikit-learn models
get_ipython().run_line_magic('run', '"../model/m6_lr.ipynb"')
get_ipython().run_line_magic('run', '"../model/m7_ann.ipynb"')
get_ipython().run_line_magic('run', '"../model/m10_rf.ipynb"')
get_ipython().run_line_magic('run', '"../model/m11_svr.ipynb"')

# PyTorch models
get_ipython().run_line_magic('run', '"../model/m8_dnn.ipynb"')
get_ipython().run_line_magic('run', '"../model/m9_rt.ipynb"')
get_ipython().run_line_magic('run', '"../model/m12_rnn.ipynb"')
get_ipython().run_line_magic('run', '"../model/m13_lstm.ipynb"')
get_ipython().run_line_magic('run', '"../model/m14_gru.ipynb"')
get_ipython().run_line_magic('run', '"../model/m15_transformer.ipynb"')

# Other frameworks
get_ipython().run_line_magic('run', '"../model/m16_prophet.ipynb"  # FB Prophet')
get_ipython().run_line_magic('run', '"../model/m17_xgb.ipynb"      # XGBoost')
get_ipython().run_line_magic('run', '"../model/m18_nbeats.ipynb"   # PyTorch Forecasting')


# # PATH

# In[ ]:


path_data_cleaned  = '../../data/'
path_result = '../../experiment_result/'


# # IMAGE CONFIG: COLOR PALETTE, FONT SETTING

# In[2]:


black = 'black'
dark_blue = '#22303D'
blue = '#2F4D67'
light_blue = '#5C7D99'
slight_blue = '#B7C9D9'
orange = '#EB932C'
platinum = '#EBE3E3'
unsw_yellow = '#ffd600'

my_figsize = (8,6)

my_font = 'arial'
font_normal ='16'
font_big = '20'
font_small = '12'

fontdict_normal = {'family': my_font,  # Set font family (e.g., 'serif', 'sans-serif', 'monospace')
                   'color':  black, # Set font color
                   'weight': 'normal',    # Set font weight (e.g., 'bold', 'normal')
                   'size': font_normal}          # Set font size


# # CROSS VALIDATION

# In[2]:


k = 10  #10 fold cross validations using blocks
n_block = k + 1 #10 for training and cross validation, 1 for hold out just in case want to test

test_pct = 1/k
train_pct = 1 - test_pct


# # FEATURE ENGINEERING

# In[ ]:


max_lag_day = 7 #lookback period in days

