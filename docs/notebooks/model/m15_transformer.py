#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train_model_m15_transformer(hyperparameter, train_df_X, train_df_y):
    """
    Train a Transformer model for point forecasting using lag and exogenous features.

    The model uses a Transformer encoder to capture temporal dependencies in lag features,
    and a fully connected layer to combine the last hidden state with exogenous features.

    Args:
        hyperparameter (dict): Dictionary containing model hyperparameters including
            seed, input_size, hidden_size, num_layers, output_size, batch_size,
            epochs, nhead, and learning_rate.
        train_df_X (pd.DataFrame): Feature matrix for training (lag + exogenous features).
        train_df_y (pd.DataFrame): Target variable for training.

    Returns:
        model (dict): Dictionary containing the trained Transformer model and associated
            hyperparameters, as well as the original training data.
    """

    # UNPACK HYPERPARAMETER
    seed = int(hyperparameter['seed'])
    input_size = int(hyperparameter['input_size'])
    hidden_size = int(hyperparameter['hidden_size'])
    num_layers = int(hyperparameter['num_layers'])
    output_size = int(hyperparameter['output_size'])
    batch_size = int(hyperparameter['batch_size'])
    epochs = int(hyperparameter['epochs'])
    nhead = int(hyperparameter['nhead'])
    learning_rate = hyperparameter['learning_rate']

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random, numpy as np, os, time
    from torch.utils.data import DataLoader, TensorDataset

    # TRANSFORMER MODEL
    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, exog_size, output_size=1):
            super(TransformerModel, self).__init__()
            # Transformer embedding
            self.embedding = nn.Linear(input_size, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=hidden_size * 2,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # Fully connected output layer
            self.fc = nn.Linear(hidden_size + exog_size, output_size)

        def forward(self, x, exogenous_data):
            x = self.embedding(x) 
            x = self.transformer_encoder(x)
            last_hidden_state = x[:, -1, :]
            combined_input = torch.cat((last_hidden_state, exogenous_data), dim=1)
            out = self.fc(combined_input)
            return out

    def train_transformer_with_minibatches(model, train_loader, epochs, learning_rate=learning_rate):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}/{epochs}]')
            start_time = time.time()
            model.train()
            batch_no = 1
            for X_lags_batch, X_exog_batch, y_batch in train_loader:
                print(f'Epoch [{epoch+1}/{epochs}] batch [{batch_no}/{len(train_loader)}]')
                batch_no += 1
                predictions = model(X_lags_batch, X_exog_batch)
                loss = criterion(predictions, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            end_time = time.time()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, time: {end_time - start_time:.2f}s')

    def set_seed(seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    # --- DATA PREP ---
    X_lags, X_exog = separate_lag_and_exogenous_features(train_df_X)
    X_lags_tensor = torch.tensor(X_lags.values, dtype=torch.float32)
    X_exog_tensor = torch.tensor(X_exog.values, dtype=torch.float32)
    y_tensor = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1)
    total_lag_features = X_lags_tensor.shape[1]
    sequence_length = total_lag_features // input_size
    exog_size = X_exog_tensor.shape[1]
    X_lags_tensor = X_lags_tensor.view(-1, sequence_length, input_size)

    # --- INIT MODEL AND DATALOADER ---
    set_seed(seed)
    transformer = TransformerModel(input_size, hidden_size, num_layers, exog_size, output_size)
    train_data = TensorDataset(X_lags_tensor, X_exog_tensor, y_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_transformer_with_minibatches(transformer, train_loader, epochs=epochs, learning_rate=learning_rate)

    model = {"transformer": transformer, 'hyperparameter': hyperparameter, "train_df_X": train_df_X, "train_df_y": train_df_y}
    return model


# In[ ]:


def produce_forecast_m15_transformer(model, train_df_X, test_df_X):
    """
    Generate forecasts for train and test sets using a trained Transformer model.

    Args:
        model (dict): Dictionary containing the trained Transformer model and hyperparameters.
        train_df_X (pd.DataFrame): Predictors of the training set (lag + exogenous features).
        test_df_X (pd.DataFrame): Predictors of the test set (lag + exogenous features).

    Returns:
        train_df_y_hat (np.ndarray): Forecast results for the training set.
        test_df_y_hat (np.ndarray): Forecast results for the test set.
    """
    transformer = model['transformer']
    hyperparameter = model['hyperparameter']
    batch_size = int(hyperparameter['batch_size'])
    input_size = int(hyperparameter['input_size'])

    def produce_forecast(transformer, X):
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
                batch_pred = transformer(batch_X_lags, batch_X_exog)
            predictions.append(batch_pred)
        return torch.cat(predictions, dim=0).detach().numpy()

    train_df_y_hat = produce_forecast(transformer, train_df_X)
    test_df_y_hat = produce_forecast(transformer, test_df_X)
    return train_df_y_hat, test_df_y_hat

