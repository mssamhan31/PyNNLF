#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORT IMPORTANT LIBRARY
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


def train_model_m10_rf(hyperparameter, train_df_X, train_df_y):
    """Train a Random Forest model for point forecasting.

    Args:
        hyperparameter (dict): Dictionary of model hyperparameters.
        train_df_X (pd.DataFrame): Feature matrix for training.
        train_df_y (pd.DataFrame): Target values for training.

    Returns:
        model (dict): Trained model containing the Random Forest under key 'rf'.
    """
    
    #UNPACK HYPERPARAMETER
    seed = int(hyperparameter['seed'])
    n_estimators = int(hyperparameter['n_estimators'])
    max_depth = int(hyperparameter['max_depth'])
    min_samples_split = int(hyperparameter['min_samples_split'])
    min_samples_leaf = int(hyperparameter['min_samples_leaf'])
    
    
    #TRAIN MODEL
    rf = RandomForestRegressor(
        n_estimators=n_estimators,       # number of trees
        max_depth=max_depth,           # maximum depth of a tree
        min_samples_split=min_samples_split,    # min samples to split a node
        min_samples_leaf=min_samples_leaf,     # min samples in a leaf
        random_state=seed
    )
    
    rf.fit(train_df_X, train_df_y) # fit the model to the training data
  
    # PACK MODEL
    model = {"rf": rf}
  

    return model


# In[ ]:


def produce_forecast_m10_rf(model, train_df_X, test_df_X):
    """Generate point forecasts on train and test sets using a trained Random Forest model.

    Args:
        model (dict): Trained model containing the Random Forest under key 'rf'.
        train_df_X (pd.DataFrame): Predictor matrix for the training set.
        test_df_X (pd.DataFrame): Predictor matrix for the test set.

    Returns:
        train_df_y_hat (pd.DataFrame): Forecasted target values for the training set.
        test_df_y_hat (pd.DataFrame): Forecasted target values for the test set.
    """
    
    # UNPACK MODEL
    rf = model['rf']
    
    # PRODUCE FORECAST
    train_df_y_hat = pd.DataFrame(rf.predict(train_df_X), index = train_df_X.index, columns = ['y_hat'])
    test_df_y_hat = pd.DataFrame(rf.predict(test_df_X), index = test_df_X.index, columns = ['y_hat'])
    
    return train_df_y_hat, test_df_y_hat

