#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORT IMPORTANT LIBRARY
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[ ]:


def train_model_m9_rt(hyperparameter, train_df_X, train_df_y):
    """Train a regression tree (DecisionTreeRegressor) for point forecasting.

    Args:
        hyperparameter (dict): Dictionary of hyperparameters (seed, max_depth, min_samples_split, min_samples_leaf, max_features).
        train_df_X (pd.DataFrame): Predictor variables for training.
        train_df_y (pd.DataFrame): Target variable for training.

    Returns:
        model (dict): Contains the trained regression tree under key 'rt'.
    """
    
    #UNPACK HYPERPARAMETER
    seed = hyperparameter['seed']
    max_depth = hyperparameter['max_depth']
    min_samples_split = hyperparameter['min_samples_split']
    min_samples_leaf = hyperparameter['min_samples_leaf']
    max_features = hyperparameter['max_features']
    
    #TRAIN MODEL
    # Initialize the regression tree model with important hyperparameters
    regressor = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        random_state = seed
    )
    
    # Train the model
    regressor.fit(train_df_X, train_df_y)
  
    # PACK MODEL
    model = {"rt": regressor}
    
    # print('I am here after training the model')
  
    return model


# In[ ]:


def produce_forecast_m9_rt(model, train_df_X, test_df_X):
    """Generate forecasts for train and test sets using a trained regression tree.

    Args:
        model (dict): Trained model containing the regression tree under key 'rt'.
        train_df_X (pd.DataFrame): Predictor variables for the training set.
        test_df_X (pd.DataFrame): Predictor variables for the test set.

    Returns:
        train_df_y_hat (pd.DataFrame): Forecasts for the training set.
        test_df_y_hat (pd.DataFrame): Forecasts for the test set.
    """
    
    # UNPACK MODEL
    regressor = model['rt']
    
    # PRODUCE FORECAST
    train_df_y_hat = pd.DataFrame(regressor.predict(train_df_X), index = train_df_X.index, columns = ['y_hat'])
    test_df_y_hat = pd.DataFrame(regressor.predict(test_df_X), index = test_df_X.index, columns = ['y_hat'])
    
    # print('I am here after training the model')
    # print('train_df_y_hat', train_df_y_hat)
    # print('test_df_y_hat', test_df_y_hat)
    
    return train_df_y_hat, test_df_y_hat

