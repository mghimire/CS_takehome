import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataimport import import_csv_parquet
from cleaning import *

# Linear Regression Model Class
class LinearRegressionModel:
    def __init__(self):
        """
        Initialize the Linear Regression model.

        Parameters: None
        """
        self.model = LinearRegression()

    def train(self, X_train, y_train, time_train):
        """
        Train the Linear Regression model on the given training data.

        Parameters:
        - X_train: pandas.DataFrame
          Training data for independent variables.
        - y_train: pandas.Series
          Training data for the target variable.
        - time_train: pandas.Series
          Training data for the time variable.
        """
        # Combine X and time features
        X_with_time = pd.concat([X_train, time_train], axis=1)
        self.model.fit(X_with_time, y_train)

    def predict(self, X_test, time_test):
        """
        Generate predictions using the trained Linear Regression model.

        Parameters:
        - X_test: pandas.DataFrame
          Test data for independent variables.
        - time_test: pandas.Series
          Test data for the time variable.

        Returns:
        - numpy.ndarray
          Array of predicted values.
        """
        # Combine X and time features
        X_with_time = pd.concat([X_test, time_test], axis=1)
        return self.model.predict(X_with_time)


# XGBoost Model Class
class XGBoostTimeSeriesModel:
    def __init__(self, params=None):
        """
        Initialize the XGBoost model for time series forecasting.

        Parameters:
        - params: dict, optional, default: None
          XGBoost hyperparameters.
        """
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'verbosity': 0
            }
        self.params = params
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the XGBoost model on the given training data.

        Parameters:
        - X_train: pandas.DataFrame
          Training data for independent variables.
        - y_train: pandas.Series
          Training data for the target variable.
        """
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Generate predictions using the trained XGBoost model.

        Parameters:
        - X_test: pandas.DataFrame
          Test data for independent variables.

        Returns:
        - numpy.ndarray
          Array of predicted values.
        """
        return self.model.predict(X_test)

# LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the LSTM model.

        Parameters:
        - input_size: int
          Number of features in the input data.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
        - x: torch.Tensor
          Input data as a PyTorch tensor.

        Returns:
        - torch.Tensor
          Output of the model.
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        output = self.fc(lstm_out)
        return output

def train_lstm_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Train the LSTM model.

    Parameters:
    - model: nn.Module
      PyTorch model to be trained.
    - train_loader: torch.utils.data.DataLoader
      DataLoader for training data.
    - criterion: nn.Module
      Loss function.
    - optimizer: torch.optim.Optimizer
      Optimizer for updating model parameters.
    - epochs: int, optional, default: 10
      Number of training epochs.
    """
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def calculate_percentage_error(actual, predicted, epsilon=1e-8):
    absolute_error = np.abs(predicted - actual)
    denominator = np.maximum(np.abs(actual), epsilon)  # Add epsilon to avoid division by zero
    percentage_error = (absolute_error / denominator) * 100
    return percentage_error
if __name__ == "__main__":

    # Test on a single file to ensure the models run fine
    train_file_path = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220103.csv.parquet'
    test_file_path = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220104.csv.parquet'
  
    # Import the data into a pandas dataframe array
    df_tr = import_csv_parquet(train_file_path)
    df_te = import_csv_parquet(test_file_path)

    # Extract y1, Xy1 and y2, Xy2 using function from cleaning
    y1_tr, Xy1_tr, time1_tr = extract(df_tr, 1)
    y2_tr, Xy2_tr, time2_tr = extract(df_tr, 2)
    y1_te, Xy1_te, time1_te = extract(df_te, 1)
    y2_te, Xy2_te, time2_te = extract(df_te, 2)

    # Handle NaN values by linear imputation
    Xy1_tr = Xy1_tr.interpolate(method='linear', axis=0).ffill().bfill()
    Xy2_tr = Xy2_tr.interpolate(method='linear', axis=0).ffill().bfill()
    Xy1_te = Xy1_te.interpolate(method='linear', axis=0).ffill().bfill()
    Xy2_te = Xy2_te.interpolate(method='linear', axis=0).ffill().bfill()

    """ # Standardize the features for LSTM
    scaler = StandardScaler()
    Xy1_tr = scaler.fit_transform(Xy1_tr)
    Xy1_te = scaler.transform(Xy1_te)
    Xy2_tr = scaler.fit_transform(Xy2_tr)
    Xy2_te = scaler.transform(Xy2_te)

    # Convert data to PyTorch tensors
    Xy1_tr_tensor = torch.tensor(Xy1_tr, dtype=torch.float32)
    Xy2_tr_tensor = torch.tensor(Xy2_tr, dtype=torch.float32)
    y1_tr_tensor = torch.tensor(y1_tr.values, dtype=torch.float32)
    y2_tr_tensor = torch.tensor(y2_tr.values, dtype=torch.float32)
    Xy1_te_tensor = torch.tensor(Xy1_te, dtype=torch.float32)
    Xy2_te_tensor = torch.tensor(Xy2_te, dtype=torch.float32)
    time1_tr_tensor = torch.tensor(time1_tr.values, dtype=torch.float32)
    time1_te_tensor = torch.tensor(time1_te.values, dtype=torch.float32)
    time2_tr_tensor = torch.tensor(time2_tr.values, dtype=torch.float32)
    time2_te_tensor = torch.tensor(time2_te.values, dtype=torch.float32) """

    linear_model_y1 = LinearRegressionModel()
    linear_model_y2 = LinearRegressionModel()

    linear_model_y1.train(Xy1_tr, y1_tr, time1_tr)
    linear_model_y2.train(Xy2_tr, y2_tr, time2_tr)
    linear_predictions_y1 = linear_model_y1.predict(Xy1_te, time1_te)
    linear_predictions_y2 = linear_model_y2.predict(Xy2_te, time2_te)

    mse_linear_y1 = mean_squared_error(y1_te, linear_predictions_y1)
    mse_linear_y2 = mean_squared_error(y2_te, linear_predictions_y2)

    print("Mean Squared Error (Linear Regression - Y1):", mse_linear_y1)
    print("Mean Squared Error (Linear Regression - Y2):", mse_linear_y2)

    # Calculate Percentage Error
    percentage_error_y1 = calculate_percentage_error(y1_te, linear_predictions_y1)
    percentage_error_y2 = calculate_percentage_error(y2_te, linear_predictions_y2)

    # Evaluate Percentage Error for Y1
    average_percentage_error_y1 = np.mean(percentage_error_y1)
    average_percentage_error_y2 = np.mean(percentage_error_y2)

    print("Average Percentage Error (Linear Regression - Y1):", average_percentage_error_y1)
    print("Average Percentage Error (Linear Regression - Y2):", average_percentage_error_y2)

    """ # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Instantiate and train ARIMA model for Y1
    arima_model_y1 = ARIMAModel(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    arima_model_y1.train(y_train)
    arima_predictions_y1 = arima_model_y1.predict(steps=len(y_test))

    # Instantiate and train XGBoost model for Y1
    xgboost_model_y1 = XGBoostTimeSeriesModel()
    xgboost_model_y1.train(X_train, y_train)
    xgboost_predictions_y1 = xgboost_model_y1.predict(X_test)

    # LSTM Model
    lstm_model_y1 = LSTMModel(input_size=X_train_tensor.shape[2])

    # Train LSTM model for Y1
    train_lstm_model(lstm_model_y1, train_loader, nn.MSELoss(), optim.Adam(lstm_model_y1.parameters()), epochs=10)

    # Make predictions using LSTM model for Y1
    lstm_predictions_y1 = lstm_model_y1.predict(X_test_tensor)

    # Reshape predictions for evaluation
    lstm_predictions_y1 = lstm_predictions_y1.detach().numpy().reshape(-1)

    # Evaluate ARIMA, XGBoost, and LSTM models for Y1
    mse_arima_y1 = mean_squared_error(y_test, arima_predictions_y1)
    mse_xgboost_y1 = mean_squared_error(y_test, xgboost_predictions_y1)
    mse_lstm_y1 = mean_squared_error(y_test, lstm_predictions_y1)

    print("Mean Squared Error (ARIMA - Y1):", mse_arima_y1)
    print("Mean Squared Error (XGBoost - Y1):", mse_xgboost_y1)
    print("Mean Squared Error (LSTM - Y1):", mse_lstm_y1) """