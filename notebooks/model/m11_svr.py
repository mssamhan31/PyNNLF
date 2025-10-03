#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


def train_model_m11_svr(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a linear model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
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
    """Create forecast at the train and test set using the trained model

    Args:
        model (dictionary): all parameters of the trained model
        train_df_X (df): predictors of train set
        test_df_X (df): predictors of test set

    Returns:
        train_df_y_hat (df) : forecast result at train set
        test_df_y_hat (df) : forecast result at test set
        
    """
    
    svr = model['svr']
    train_df_y_hat = svr.predict(train_df_X)
    test_df_y_hat = svr.predict(test_df_X)

    return train_df_y_hat, test_df_y_hat

