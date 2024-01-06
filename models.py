# models.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

class LinearRegressionModel:
    def __init__(self):
        self.model = nn.Linear(in_features=375, out_features=1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, X_train, y_train, epochs=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

    def predict(self, X_test):
        with torch.no_grad():
            return self.model(X_test).numpy()


class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingRegressor()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        output = self.fc(lstm_out)
        return output

def train_lstm_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
  
  # Extract dependent variables Y1 and Y2
  y1 = df['Y1'].values.reshape(-1, 1)
  y2 = df['Y2'].values.reshape(-1, 1)
  
  # Extract independent variables X1 to X375
  X = df.drop(['time', 'Y1', 'Y2'], axis=1).values
  
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)
  
  # Standardize the features
  scaler = StandardScaler()
  X_train_std = scaler.fit_transform(X_train)
  X_test_std = scaler.transform(X_test)
  
  # Convert data to PyTorch tensors
  X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  X_test_tensor = torch.tensor(X_test_std, dtype=torch.float32)
  
  # Create DataLoader for training
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  
  # Instantiate and train Linear Regression model
  linear_regression_model = LinearRegressionModel()
  linear_regression_model.train(X_train_tensor, y_train_tensor, epochs=100)
  
  # Make predictions using Linear Regression model
  linear_regression_predictions = linear_regression_model.predict(X_test_tensor)
  
  # Calculate and print Mean Squared Error for Linear Regression
  mse_linear_regression = mean_squared_error(y_test, linear_regression_predictions)
  print("Mean Squared Error (Linear Regression):", mse_linear_regression)

  # Repeat the process for Gradient Boosting
  gradient_boosting_model = GradientBoostingModel()
  gradient_boosting_model.train(X_train, y_train)
  gradient_boosting_predictions = gradient_boosting_model.predict(X_test)
  mse_gradient_boosting = mean_squared_error(y_test, gradient_boosting_predictions)
  print("Mean Squared Error (Gradient Boosting):", mse_gradient_boosting)

    
  # Instantiate and train LSTM model
  lstm_model = LSTMModel(input_size=X_train_tensor.shape[2])
  criterion = nn.MSELoss()
  optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
  train_lstm_model(lstm_model, train_loader, criterion, optimizer, epochs=10)
  
  # Make predictions using LSTM model
  lstm_predictions = lstm_model(X_test_tensor).detach().numpy()
  
  # Calculate and print Mean Squared Error for LSTM
  mse_lstm = mean_squared_error(y_test, lstm_predictions)
  print("Mean Squared Error (LSTM):", mse_lstm)
