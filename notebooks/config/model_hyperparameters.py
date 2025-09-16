#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[1]:


m1 = 'm1_naive' #custom
m2 = 'm2_snaive' #custom
m3 = 'm3_ets' #statsmodel
m4 = 'm4_arima' #statsmodel
m5 = 'm5_sarima' #statsmodel
m6 = 'm6_lr' #scikit-learn
m7 = 'm7_ann' #scikit-learn
m8 = 'm8_dnn' #pytorch
m9 = 'm9_rt' #pytorch
m10 = 'm10_rf' #scikit-learn
m11 = 'm11_svr' #scikit-learn
m12 = 'm12_rnn' #pytorch
m13 = 'm13_lstm' #pytorch
m14 = 'm14_gru' #pytorch
m15 = 'm15_transformer' #pytorch
m16 = 'm16_prophet' #fb prophet
m17 = 'm17_xgb' #xgboost
m18 = 'm18_nbeats' #pytorch forecasting


# # Model 1. Naive

# In[11]:


# Hyperparameter List:
# version
m1_hp_table = [
    {
        "hp_no": "hp1",
        "hyperparameter_1": "No hyperparameter",
    }
]

m1_hp_table = pd.DataFrame(m1_hp_table)
m1_hp_table.set_index('hp_no', inplace=True)


# # Model 2. Seasonal Naive

# In[ ]:


# Hyperparameter List:
# We can also add initial states or damping 

# version

m2_hp_table = [
    {
        "hp_no": "hp1",
        "days": 1, #1 day behind
    },
    {
        "hp_no": "hp2",
        "days": 7, #a week behind
    }
]

m2_hp_table = pd.DataFrame(m2_hp_table)
m2_hp_table.set_index('hp_no', inplace=True)


# # Model 3. Exponential Smoothing

# Remember to inlcude seed!

# In[ ]:


# Hyperparameter List:
# We don't use seasonal period 7 days because it failed to converge (tried 250109)
m3_hp_table = [
    {
        "hp_no": "hp1",
        "trend": False,
        "damped_trend": False,
        "seasonal_periods_days": None
    },
    {
        "hp_no": "hp2",
        "trend": False,
        "damped_trend": False,
        "seasonal_periods_days": 1
    },
    {
        "hp_no": "hp3",
        "trend": "add",
        "damped_trend": False,
        "seasonal_periods_days": None
    },
]

m3_hp_table = pd.DataFrame(m3_hp_table)
m3_hp_table.set_index('hp_no', inplace=True)


# # Model 4. ARIMA

# In[ ]:


# Hyperparameter List:
m4_hp_table = [
    {
        "hp_no": "hp1",
        "p": 1,
        "d": 0,
        "q": 1
    },
    {
        "hp_no": "hp2",
        "p": 1,
        "d": 1,
        "q": 1
    }
]

m4_hp_table = pd.DataFrame(m4_hp_table)
m4_hp_table.set_index('hp_no', inplace=True)


# # Model 5. SARIMA

# In[ ]:


# Hyperparameter List:
# this model is not used because it is very slow to converge somehow
m5_hp_table = [
    {
        "hp_no": "hp1",
        "p": 1,
        "d": 0,
        "q": 1,
        "P": 0,
        "D": 1,
        "Q": 0,
        "seasonal_period_days": 1
    },
    {
        "hp_no": "hp2",
        "p": 1,
        "d": 0,
        "q": 1,
        "P": 0,
        "D": 1,
        "Q": 0,
        "seasonal_period_days": 7
    }
]

m5_hp_table = pd.DataFrame(m5_hp_table)
m5_hp_table.set_index('hp_no', inplace=True)


# # Model 6. Linear Regression

# In[ ]:


# no need to set up seed because deterministic
m6_hp_table = [
    {
        "hp_no": "hp1",
        "num_features": 50,
    },
]

m6_hp_table = pd.DataFrame(m6_hp_table)
m6_hp_table.set_index('hp_no', inplace=True)


# # Model 7. ANN

# Remember to inlcude seed!
# Include n of epoch, batch size, optimiser used, learning rate, regularization technique, early stopping etc.

# In[ ]:


m7_hp_table = [
    {
        "hp_no": "hp1",
        "seed": 99, # we will use the same seed for reproducibility
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.001,
        "solver" : "adam",
        "epochs" : 500
    },
    {
        "hp_no": "hp2",
        "seed": 99, # we will use the same seed for reproducibility
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.01,
        "solver" : "adam",
        "epochs" : 500
    },
]

m7_hp_table = pd.DataFrame(m7_hp_table)
m7_hp_table.set_index('hp_no', inplace=True)


# # Model 8. DNN

# In[ ]:


m8_hp_table = [
    {
        "hp_no": "hp1",
        "seed": 1, # we will use the same seed for reproducibility #I tried 99 but it failed! It somehow gives non negative result.
        "n_hidden" : 3, #number of hidden layers
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.001,
        "solver" : "adam",
        "epochs" : 500
    },
    {
        "hp_no": "hp2",
        "seed": 1, # we will use the same seed for reproducibility
        "n_hidden" : 3, #number of hidden layers
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.01,
        "solver" : "adam",
        "epochs" : 500
    },
    {
        "hp_no": "hp3",
        "seed": 1, # we will use the same seed for reproducibility
        "n_hidden" : 4, #number of hidden layers
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.01,
        "solver" : "adam",
        "epochs" : 500
    },
]

m8_hp_table = pd.DataFrame(m8_hp_table)
m8_hp_table.set_index('hp_no', inplace=True)


# # Model 9. Regression Tree

# In[ ]:


m9_hp_table = [
    {
        "hp_no": "hp1",
        "seed": 1, 
        "max_depth" : 3,
        "min_samples_split" : 2,
        "min_samples_leaf" : 1,
        "max_features" : "sqrt",
    },
    {
        "hp_no": "hp2",
        "seed": 1, 
        "max_depth" : 30,
        "min_samples_split" : 2,
        "min_samples_leaf" : 1,
        "max_features" : "sqrt",
    },
    {
        "hp_no": "hp3",
        "seed": 1, 
        "max_depth" : 15,
        "min_samples_split" : 2,
        "min_samples_leaf" : 1,
        "max_features" : "sqrt",
    }
]

m9_hp_table = pd.DataFrame(m9_hp_table)
m9_hp_table.set_index('hp_no', inplace=True)


# # Model 10. Random Forest

# In[ ]:


m10_hp_table = [
    {
        "hp_no": "hp1",
        "seed": 1, 
        "n_estimators" : 100,
        "max_depth" : 3,
        "min_samples_split" : 2,
        "min_samples_leaf" : 1
    },
    {
        "hp_no": "hp2",
        "seed": 1, 
        "n_estimators" : 50,
        "max_depth" : 3,
        "min_samples_split" : 2,
        "min_samples_leaf" : 1
    }
]

m10_hp_table = pd.DataFrame(m10_hp_table)
m10_hp_table.set_index('hp_no', inplace=True)


# # Model 11. SVR

# In[ ]:


m11_hp_table = [
    {
        "hp_no": "hp1", #this is only for testing, with epochs = 2 just to ensure code is working.
        "seed": 1,              # Seed for reproducibility
        "kernel": 'rbf', # kernel
        "C": 100, # regularization strength, higher = less regularization
     "gamma": 0.001, # kernel coefficient, higher = more complex
        "epsilon": 0.3 #margin of tolerance 
    }
]


m11_hp_table = pd.DataFrame(m11_hp_table)
m11_hp_table.set_index('hp_no', inplace=True)


# # Model 12. RNN

# In[ ]:


m12_hp_table = [
    {
        "hp_no": "hp1", #this is only for testing, with epochs = 2 just to ensure code is working.
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,      # RNN units
        "num_layers": 1,        # RNN layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "epochs": 2,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    },
    {
        "hp_no": "hp2",
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,     # RNN units
        "num_layers": 1,        # RNN layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "epochs": 100,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    }
]


m12_hp_table = pd.DataFrame(m12_hp_table)
m12_hp_table.set_index('hp_no', inplace=True)


# # Model 13. LSTM

# In[ ]:


m13_hp_table = [
    {
        "hp_no": "hp1", #this is only for testing, with epochs = 2 just to ensure code is working.
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,      # LSTM units
        "num_layers": 1,        # LSTM layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "epochs": 2,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    },
    {
        "hp_no": "hp2",
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,     # LSTM units
        "num_layers": 1,        # LSTM layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "epochs": 100,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    }
]


m13_hp_table = pd.DataFrame(m13_hp_table)
m13_hp_table.set_index('hp_no', inplace=True)


# # Model 14. GRU

# In[ ]:


m14_hp_table = [
    {
        "hp_no": "hp1", #this is only for testing, with epochs = 2 just to ensure code is working.
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,      # GRU units
        "num_layers": 1,        # GRU layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "epochs": 2,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    },
    {
        "hp_no": "hp2",
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,     # GRU units
        "num_layers": 1,        # GRU layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "epochs": 100,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    }
]


m14_hp_table = pd.DataFrame(m14_hp_table)
m14_hp_table.set_index('hp_no', inplace=True)


# # Model 15. TRANSFORMER

# In[ ]:


m15_hp_table = [
    {
        "hp_no": "hp1", #this is only for testing, with epochs = 2 just to ensure code is working.
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,      # transformer units
        "num_layers": 1,        # transformer layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "nhead" : 4,           # Number of attention heads
        "epochs": 2,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    },
    {
        "hp_no": "hp2",
        "seed": 1,              # Seed for reproducibility
        "input_size": 1,        # One feature per timestep (e.g., lag)
        "hidden_size": 64,     # transformer units
        "num_layers": 1,        # transformer layers
        "output_size": 1,       # One-step prediction
        "batch_size": 4096,     # Training batch size
        "nhead" : 4,           # Number of attention heads
        "epochs": 100,            # Training epochs
        "learning_rate": 0.001  # Optimizer learning rate
    }
]


m15_hp_table = pd.DataFrame(m15_hp_table)
m15_hp_table.set_index('hp_no', inplace=True)


# # Model 16. PROPHET

# In[ ]:


m16_hp_table = [
    {
        "hp_no": "hp1",
        "seed": 1,                      # Seed for reproducibility
        "seasonality_prior_scale": 10,           # Strength of seasonality (higher = stronger seasonality)
        "seasonality_mode": "additive",         # Type of seasonality: 'additive' or 'multiplicative'
        "weekly_seasonality": True,             # Enable weekly seasonality
        "daily_seasonality": True,              # Enable daily seasonality
        "growth": "linear"                      # Model type: 'linear' or 'logistic'
    }
]

m16_hp_table = pd.DataFrame(m16_hp_table)
m16_hp_table.set_index('hp_no', inplace=True)


# # Model 17. XGBOOST

# In[ ]:


m17_hp_table = [
    {
        "hp_no": "hp1",
        'xgb_seed': 1,             # Seed for reproducibility
        'n_estimators': 200,          # Number of trees
        'learning_rate': 0.1,         # Step size for gradient descent
        'max_depth': 6,               # Maximum depth of each tree
        'subsample': 0.8,             # Fraction of training data used for each tree
        'colsample_bytree': 0.8       # Fraction of features used for each tree
        # this seems to result in overfittting
    },
    {
        "hp_no": "hp2",
        'xgb_seed': 1,             # Seed for reproducibility
        'n_estimators': 100,          # Number of trees
        'learning_rate': 0.1,         # Step size for gradient descent
        'max_depth': 3,               # Maximum depth of each tree
        'subsample': 0.8,             # Fraction of training data used for each tree
        'colsample_bytree': 0.8       # Fraction of features used for each tree
    }
]

m17_hp_table = pd.DataFrame(m17_hp_table)
m17_hp_table.set_index('hp_no', inplace=True)


# # Model 18. NBEATS

# In[ ]:


m18_hp_table = [
    {
        "hp_no": "hp1", #this is only for testing, with epochs = 2 just to ensure code is working.
        "seed": 1,              # Seed for reproducibility
        "hidden_size": 64,      # NBeats units
        "num_layers": 1,        # NBeats layers
        "num_blocks": 3,        # number of blocks in N-BEATS
        "output_size": 1,       # One-step prediction
        "epochs": 2,            # Training epochs
        "lr": 0.001  # Optimizer learning rate
    },
    {
        "hp_no": "hp2",
        "seed": 1,              # Seed for reproducibility
        "hidden_size": 64,     # NBeats units
        "num_layers": 1,        # NBeats layers
        "num_blocks": 3,        # number of blocks in N-BEATS
        "output_size": 1,       # One-step prediction
        "epochs": 100,            # Training epochs
        "lr": 0.001  # Optimizer learning rate
    }
]


m18_hp_table = pd.DataFrame(m18_hp_table)
m18_hp_table.set_index('hp_no', inplace=True)

