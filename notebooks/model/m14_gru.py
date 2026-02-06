#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train_model_m14_gru(hyperparameter, train_df_X, train_df_y):
    """
    Train a GRU-based model for point forecasting.

    The GRU captures temporal patterns from lag features, and a fully connected
    layer integrates the last hidden state with exogenous variables to produce
    the final prediction.

    Args:
        hyperparameter (dict): Dictionary of hyperparameters, including:
            - 'seed': random seed for reproducibility
            - 'input_size': number of input features per time step
            - 'hidden_size': number of hidden units in GRU
            - 'num_layers': number of GRU layers
            - 'output_size': number of outputs
            - 'batch_size': minibatch size
            - 'epochs': number of training epochs
            - 'learning_rate': optimizer learning rate
        train_df_X (pd.DataFrame): DataFrame of input features (lag + exogenous).
        train_df_y (pd.DataFrame): DataFrame of target values.

    Returns:
        model (dict): Dictionary containing:
            - 'gru': trained GRU model
            - 'hyperparameter': hyperparameters used for training
            - 'train_df_X': training features
            - 'train_df_y': training target
    """

    # UNPACK HYPERPARAMETER
    seed = int(hyperparameter['seed'])
    input_size = int(hyperparameter['input_size'])
    hidden_size = int(hyperparameter['hidden_size'])
    num_layers = int(hyperparameter['num_layers'])
    output_size = int(hyperparameter['output_size'])
    batch_size = int(hyperparameter['batch_size'])
    epochs = int(hyperparameter['epochs'])
    learning_rate = hyperparameter['learning_rate']

    # DEFINE MODEL
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, exog_size, output_size=1):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size + exog_size, output_size)

        def forward(self, x, exogenous_data):
            out, h_n = self.gru(x)
            last_hidden_state = out[:, -1, :]
            combined_input = torch.cat((last_hidden_state, exogenous_data), dim=1)
            out = self.fc(combined_input)
            return out

    def train_gru_with_minibatches(model, train_loader, epochs, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}/{epochs}]')
            start_time = time.time()
            model.train()
            batch_no = 1
            for X_lags_batch, X_exog_batch, y_batch in train_loader:
                print(f'Epoch [{epoch+1}/{epochs}] and batch [{batch_no}/{len(train_loader)}]')
                batch_no += 1

                predictions = model(X_lags_batch, X_exog_batch)
                loss = criterion(predictions, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, time taken: {time.time() - start_time:.2f}s')

    def set_seed(seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    # PREPARE TRAIN DATA
    X_lags, X_exog = separate_lag_and_exogenous_features(train_df_X)
    X_lags_tensor = torch.tensor(X_lags.values, dtype=torch.float32)
    X_exog_tensor = torch.tensor(X_exog.values, dtype=torch.float32)
    y_tensor = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1)

    total_lag_features = X_lags_tensor.shape[1]
    sequence_length = total_lag_features // input_size
    exog_size = X_exog_tensor.shape[1]

    X_lags_tensor = X_lags_tensor.view(-1, sequence_length, input_size)

    # INITIALIZE MODEL + DATALOADER
    set_seed(seed=seed)
    gru = GRUModel(input_size, hidden_size, num_layers, exog_size, output_size)
    train_data = TensorDataset(X_lags_tensor, X_exog_tensor, y_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # TRAIN MODEL
    train_gru_with_minibatches(gru, train_loader, epochs=epochs, learning_rate=learning_rate)

    # PACK MODEL
    model = {"gru": gru, 'hyperparameter': hyperparameter, "train_df_X": train_df_X, "train_df_y": train_df_y}
    return model


# In[ ]:


def produce_forecast_m14_gru(model, train_df_X, test_df_X):
    """
    Generate forecasts using a trained GRU model for both train and test sets.

    The GRU model processes lag features to capture temporal patterns and 
    combines them with exogenous variables via a fully connected layer to produce 
    the predictions. Minibatching is used to avoid memory issues.

    Args:
        model (dict): Dictionary containing the trained GRU model and hyperparameters.
        train_df_X (pd.DataFrame): Input features for the training set (lag + exogenous).
        test_df_X (pd.DataFrame): Input features for the test set (lag + exogenous).

    Returns:
        train_df_y_hat (np.ndarray): Forecasted values for the training set.
        test_df_y_hat (np.ndarray): Forecasted values for the test set.
    """
    
    gru = model['gru']
    hyperparameter = model['hyperparameter']
    input_size = int(hyperparameter['input_size'])
    batch_size = int(hyperparameter['batch_size'])

    def produce_forecast(gru, X):
        X_lags, X_exog = separate_lag_and_exogenous_features(X)
        X_lags_tensor = torch.tensor(X_lags.values, dtype=torch.float32)
        X_exog_tensor = torch.tensor(X_exog.values, dtype=torch.float32)

        total_lag_features = X_lags_tensor.shape[1]
        sequence_length = total_lag_features // input_size
        X_lags_tensor = X_lags_tensor.view(-1, sequence_length, input_size)

        predictions = []
        for i in range(0, len(X_lags_tensor), batch_size):
            batch_X_lags = X_lags_tensor[i:i+batch_size]
            batch_X_exog = X_exog_tensor[i:i+batch_size]
            with torch.no_grad():
                batch_pred = gru(batch_X_lags, batch_X_exog)
            predictions.append(batch_pred)
        return torch.cat(predictions, dim=0).detach().numpy()

    train_df_y_hat = produce_forecast(gru, train_df_X)
    test_df_y_hat = produce_forecast(gru, test_df_X)
    return train_df_y_hat, test_df_y_hat

