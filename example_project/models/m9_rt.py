# IMPORT IMPORTANT LIBRARY

"""Regression tree net load forecasting model.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: fit a scikit-learn decision tree regressor on the feature matrix.
"""

from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def train_model_m9_rt(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a regression tree model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
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
    
  
    return model


def produce_forecast_m9_rt(model, train_df_X, test_df_X):
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
    regressor = model['rt']
    
    # PRODUCE FORECAST
    train_df_y_hat = pd.DataFrame(regressor.predict(train_df_X), index = train_df_X.index, columns = ['y_hat'])
    test_df_y_hat = pd.DataFrame(regressor.predict(test_df_X), index = test_df_X.index, columns = ['y_hat'])
    
    
    return train_df_y_hat, test_df_y_hat

