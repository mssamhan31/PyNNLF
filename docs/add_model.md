## Model Structure

All models are stored in the `model` folder. Each model has a dedicated `.ipynb` file, making it easy to navigate and understand its implementation.

Each model typically includes two functions: one for training and one for generating forecasts. Tasks such as train-test splitting, 10-fold cross-validation, evaluation, and plotting are handled in `general_functions.ipynb`, keeping model files focused solely on model-specific code.

## Training Function

The training function trains the model using the training set predictors and target values. It returns a single object, `model`, which contains the trained model.

## Testing Function
The testing function uses the trained model along with the training and testing predictors. It returns two DataFrames: forecasted values for both the training and testing sets.

Evaluation and plotting are managed by `notebooks/config/general_functions.ipynb`.

## How to Add a Model

To add a new model, follow these three steps:

### 1. Create a new model file in the `notebooks/model/` folder

For example, to add a model named `new_model`, create a file called `m19_new_model.ipynb`. Define the training and testing functions in this file, named `train_model_m19_new_model` and `produce_forecast_model_m19_new_model`.

You can refer to existing models for examples, such as the ANN model:
```
def train_model_m7_ann(hyperparameter, train_df_X, train_df_y):
    ''' Train and test an ANN model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
    #UNPACK HYPERPARAMETER

    # Set random seed for reproducibility
    def set_seed(seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    seed = int(hyperparameter['seed'])

    hidden_size = hyperparameter['hidden_size']
    activation_function = hyperparameter['activation_function']
    learning_rate = hyperparameter['learning_rate']
    # learning_rate = 0.001
    solver = hyperparameter['solver']
    epochs = hyperparameter['epochs']
    
    # Use proper format for X and y
    X = torch.tensor(train_df_X.values, dtype=torch.float32)
    y = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1) 
    
    # Define the ANN model
    class ANNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(ANNModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()  # Activation function

        def forward(self, x):
            x = self.fc1(x)
            if activation_function == 'relu':
                x = self.relu(x)
            elif activation_function == 'sigmoid':
                x = torch.sigmoid(x)
            else:
                x = torch.tanh(x)
            x = self.fc2(x)
            return x
        
    # Model initialization
    input_size = X.shape[1]
    output_size = y.shape[1]
    
    set_seed(seed)
    
    model_ann = ANNModel(input_size, hidden_size, output_size)
    if solver == 'adam':
        optimizer = optim.Adam(model_ann.parameters(), lr=learning_rate)
    elif solver == 'sgd':
        optimizer = optim.SGD(model_ann.parameters(), lr=learning_rate)
    else:
        raise ValueError('Solver not found')
    
    # Loss function
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    
    #TRAIN MODEL
    # Training loop
    for epoch in range(epochs):
        model_ann.train()
        
        # Forward pass
        output = model_ann(X)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
  
    # PACK MODEL
    model = {"model_ann": model_ann}
  

    return model
```

```
def produce_forecast_m7_ann(model, train_df_X, test_df_X):
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
    model_ann = model["model_ann"]

    # PREPARE FORMAT
    train_df_X_tensor = torch.tensor(train_df_X.values, dtype=torch.float32)
    test_df_X_tensor = torch.tensor(test_df_X.values, dtype=torch.float32)

    # PRODUCE FORECAST
    # Switch model to evaluation mode for inference
    model_ann.eval()

    # TRAIN SET FORECAST
    with torch.no_grad():  # Disable gradient calculation to save memory
        train_df_y_hat_tensor = model_ann(train_df_X_tensor)

    # TEST SET FORECAST
    with torch.no_grad():  # Disable gradient calculation to save memory
        test_df_y_hat_tensor = model_ann(test_df_X_tensor)
        
    # Create DataFrames of result
    train_df_y_hat = pd.DataFrame(train_df_y_hat_tensor, index=train_df_X.index, columns=['y_hat'])
    test_df_y_hat = pd.DataFrame(test_df_y_hat_tensor, index=test_df_X.index, columns=['y_hat'])
    
    return train_df_y_hat, test_df_y_hat
```

You can add any model you’ve developed or proposed, as long as it can be trained using the training set features and target values. The next steps are straightforward.

### 2. Update `train_model` function in `notebooks/config/general_functions.ipynb` file
This file contains utility functions, including train_model, which dispatches training based on the selected model.

Add a new condition like:
```
elif model_name == 'm19_new_model':
        model = train_model_m19_new_model(hyperparameter, train_df_X, train_df_y)
```

### 3. Update `produce_forecast` function in `notebooks/config/general_functions.ipynb` file
Similarly, update the `produce_forecast` function by adding:
```
elif model_name == 'm19_new_model':
        train_df_y_hat, test_df_y_hat = produce_forecast_m19_new_model(model, train_df_X, test_df_X)
```