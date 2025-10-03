#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORT IMPORTANT LIBRARY
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


def train_model_m10_rf(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a random forest model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
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
    """Create forecast at the train and test set using the trained model

    Args:
        model (dictionary): all parameters of the trained model
        train_df_X (df): predictors of train set
        test_df_X (df): predictors of test set

    Returns:
        train_df_y_hat (df) : forecast result at train set
        test_df_y_hat (df) : forecast result at test set
        
    """
    
    # UNPACK MODEL
    rf = model['rf']
    
    # PRODUCE FORECAST
    train_df_y_hat = pd.DataFrame(rf.predict(train_df_X), index = train_df_X.index, columns = ['y_hat'])
    test_df_y_hat = pd.DataFrame(rf.predict(test_df_X), index = test_df_X.index, columns = ['y_hat'])
    
    return train_df_y_hat, test_df_y_hat

