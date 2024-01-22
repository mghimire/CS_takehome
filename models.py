import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import xgboost as xgb

import torch
import torch.nn as nn

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

    def train(self, X_train, y_train):
        """
        Train the Linear Regression model on the given training data.

        Parameters:
        - X_train: pandas.DataFrame
          Training data for independent variables.
        - y_train: pandas.Series
          Training data for the target variable.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Generate predictions using the trained Linear Regression model.

        Parameters:
        - X_test: pandas.DataFrame
          Test data for independent variables.

        Returns:
        - numpy.ndarray
          Array of predicted values.
        """
        return self.model.predict(X_test)


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

# Modified Neural Network Model Class
class FeedforwardModel(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the Feedforward (Standard Neural Network) model.

        Parameters:
        - input_size: int
          Number of features in the input data.
        """
        super(FeedforwardModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=10)  # Adjust output features as needed
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=10, out_features=2)  # Adjust output features as needed

    def forward(self, x):
        """
        Forward pass through the Feedforward model.

        Parameters:
        - x: torch.Tensor
          Input data as a PyTorch tensor.

        Returns:
        - torch.Tensor
          Output of the model.
        """
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output

    def train_feedforward_model(self, train_loader, criterion, optimizer, epochs=10):
        """
        Train the Feedforward model.

        Parameters:
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
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

def RMSE(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    # Avoid division by zero
    mask = actual != 0
    return np.sqrt(((actual - predicted)**2).mean())

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
    RMSE_y1 = RMSE(y1_te, linear_predictions_y1)
    RMSE_y2 = RMSE(y2_te, linear_predictions_y2)

    # Evaluate Percentage Error for Y1
    root_mean_squared_error_y1 = RMSE_y1
    root_mean_squared_error_y2 = RMSE_y2

    print("Root Mean Squared Error (Y1):", root_mean_squared_error_y1)
    print("Root Mean Squared Error (Y2):", root_mean_squared_error_y2)

    """ # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


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