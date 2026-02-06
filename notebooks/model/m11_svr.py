#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train_model_m11_svr(hyperparameter, train_df_X, train_df_y):
    """Train a Support Vector Regression (SVR) model for point forecasting.

    Args:
        hyperparameter (dict): SVR hyperparameters including 'kernel', 'C', 'gamma', and 'epsilon'.
        train_df_X (pd.DataFrame): Predictor matrix for training.
        train_df_y (pd.DataFrame or pd.Series): Target values for training.

    Returns:
        model (dict): Trained model containing the SVR object under key 'svr'.
    """
    
    from sklearn.svm import SVR
    
    #UNPACK HYPERPARAMETER
    seed = hyperparameter['seed'] #seem we can't use this using sklearn
    kernel = hyperparameter['kernel']
    C = hyperparameter['C']
    gamma = hyperparameter['gamma']
    epsilon = hyperparameter['epsilon']
        
    #TRAIN MODEL
    train_df_y = train_df_y.values.ravel()  # Flatten the target array if necessary
    svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    svr.fit(train_df_X, train_df_y)
  
    # PACK MODEL
    model = {"svr": svr}
  

    return model


# In[ ]:


def produce_forecast_m11_svr(model, train_df_X, test_df_X):
    """Generate forecasts using a trained SVR model.

    Args:
        model (dict): Trained model containing the SVR object under key 'svr'.
        train_df_X (pd.DataFrame): Predictor matrix for the training set.
        test_df_X (pd.DataFrame): Predictor matrix for the test set.

    Returns:
        train_df_y_hat (np.ndarray): Forecasted values for the training set.
        test_df_y_hat (np.ndarray): Forecasted values for the test set.
    """
    
    svr = model['svr']
    train_df_y_hat = svr.predict(train_df_X)
    test_df_y_hat = svr.predict(test_df_X)

    return train_df_y_hat, test_df_y_hat

