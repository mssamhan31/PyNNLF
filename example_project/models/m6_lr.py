"""Linear regression net load forecasting model with feature selection.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: select the k best features by univariate scoring, then fit ordinary least squares.
"""

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

def train_model_m6_lr(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a linear model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (dictionary) : trained model with all features
    '''
    
    #UNPACK HYPERPARAMETER
    num_feature = int(hyperparameter['num_features'])
    
    # FEATURE SELECTOR
    def select_features(train_df_X, train_df_y, num_feature):
        ''' Make model to select K best feature. 
            
        Args:
            train_df_X (df) : features matrix for training
            train_df_y (df) : target matrix for training
        
        Returns:
            fs_lr (model) : feature selector
        '''
        
        train_df_y = train_df_y.values.ravel()
        fs_lr = SelectKBest(f_regression, k = num_feature)
        fs_lr.fit(train_df_X, train_df_y)
        
        return fs_lr

    fs_lr = select_features(train_df_X, train_df_y, num_feature)
    
    #TRAIN MODEL
    train_df_X = fs_lr.transform(train_df_X)
    m06_lr = LinearRegression()
    m06_lr.fit(train_df_X, train_df_y)
  
    # PACK MODEL
    model = {"feature_selector": fs_lr, "regression_model": m06_lr}    

    return model

def produce_forecast_m6_lr(model, train_df_X, test_df_X):
    """Create forecast at the train and test set using the trained model

    Args:
        model (dictionary): all parameters of the trained model
        train_df_X (df): predictors of train set
        test_df_X (df): predictors of test set

    Returns:
        train_df_y_hat (df) : forecast result at train set
        test_df_y_hat (df) : forecast result at test set
        
    """
    fs_lr = model['feature_selector']
    m06_lr = model['regression_model']
    
    # SELECT K BEST FEATURES
    train_df_X = fs_lr.transform(train_df_X)
    test_df_X = fs_lr.transform(test_df_X)
    
    # PRODUCE FORECAST
    train_df_y_hat = m06_lr.predict(train_df_X)
    test_df_y_hat = m06_lr.predict(test_df_X)
    
    return train_df_y_hat, test_df_y_hat

