import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from dataimport import *
from cleaning import *

def lasso_feature_selection(X, y):
    # Initialize Lasso regression model with time-series-split cross-validated alpha selection
    tscv = TimeSeriesSplit(n_splits=5)
    lasso_model = LassoCV(cv=tscv)

    # Standardize the features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    lasso_model.fit(X_std, y)

    # Display the selected features and their corresponding coefficients
    selected_features = pd.Series(lasso_model.coef_, index=X.columns)
    selected_features = selected_features[selected_features != 0]

    return selected_features.index.tolist()

def ridge_feature_selection(X, y):
    # Initialize Ridge regression model with time-series-split cross-validated alpha selection
    tscv = TimeSeriesSplit(n_splits=5)
    ridge_model = RidgeCV(cv=tscv)

    # Standardize the features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    lasso_model.fit(X_std, y)

    # Display the selected features and their corresponding coefficients
    selected_features = pd.Series(lasso_model.coef_, index=X.columns)
    selected_features = selected_features[selected_features != 0]

    return selected_features.index.tolist()

def elastic_net_feature_selection(X, y):
    # Set cross-validation based on time series splits
    tscv = TimeSeriesSplit(n_splits=5)

    # Standardize the features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Build an Elastic Net model with cross-validated alpha selection
    elastic_net_model = ElasticNetCV(cv=tscv)
    elastic_net_model.fit(X_std, y)

    # Display the selected features and their corresponding coefficients
    selected_features = pd.Series(elastic_net_model.coef_, index=X.columns)
    selected_features = selected_features[selected_features != 0]

    return selected_features.index.tolist()


def stepwise_regression_feature_selection(X, y):
    # Add a constant column to the features for the intercept term
    X_with_intercept = sm.add_constant(X)

    # Split the data into training and testing sets based on time
    split_time = int(len(X) * 0.8)  # Adjust the split ratio as needed
    X_train, X_test = X_with_intercept.iloc[:split_time, :], X_with_intercept.iloc[split_time:, :]
    y_train, y_test = y.iloc[:split_time], y.iloc[split_time:]

    # Perform stepwise regression using statsmodels
    stepwise_model = sm.OLS(y_train, X_train).fit()
    
    # Display summary statistics
    print(stepwise_model.summary())

    # Get the most relevant features based on p-values
    selected_features = stepwise_model.pvalues[1:].idxmin()

    return selected_features

if __name__ == "__main__":

  filepath = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220103.csv.parquet'
  df = import_csv_parquet(filepath)

  df = removebadval(df)
  df = skipbadrow(df)

  # Sort the DataFrame by the time column for time series data
  df.sort_values('time', inplace=True)
  
  # Separate df into dfy1 and dfy2 based on the 'Q1' and 'Q2' values for
  # appropriate calibration (remove rows with 'Qi' < 1 for dfyi for i = 1, 2)

  dfy1 = df[df['Q1'] >= 0.9999]
  dfy2 = df[df['Q2'] >= 0.9999]

  # Extract dependent variables Y1 and Y2
  y1 = dfy1['Y1']
  y2 = dfy2['Y2']
  
  # Extract independent variables X1 to X375
  Xy1 = dfy1.drop(['time', 'sym', 'exch', 'Q1', 'Q2', 'Y1', 'Y2'], axis=1)
  Xy2 = dfy2.drop(['time', 'sym', 'exch', 'Q1', 'Q2', 'Y1', 'Y2'], axis=1)

  # Handle NaN values by linear imputation
  Xy1_imputed = Xy1.interpolate(method='linear', axis=0).ffill().bfill()
  Xy2_imputed = Xy2.interpolate(method='linear', axis=0).ffill().bfill()

  # Perform Lasso feature selection for Y1
  selected_features_y1_lasso = lasso_feature_selection(Xy1_imputed, y1)
  print("Lasso Selected Features for Y1:", selected_features_y1_lasso)
  
  # Perform Lasso feature selection for Y2
  selected_features_y2_lasso = lasso_feature_selection(Xy2_imputed, y2)
  print("Lasso Selected Features for Y2:", selected_features_y2_lasso)

  # Perform Elastic Net feature selection for Y1
  #selected_features_y1_elastic_net = elastic_net_feature_selection(X_imputed, y1)
  #print("Elastic Net Selected Features for Y1:", selected_features_y1_elastic_net)
  
  # Perform Elastic Net feature selection for Y2
  #selected_features_y2_elastic_net = elastic_net_feature_selection(X_imputed, y2)
  #print("Elastic Net Selected Features for Y2:", selected_features_y2_elastic_net)
  
  # Perform Stepwise Regression feature selection for Y1
  selected_feature_y1_stepwise = stepwise_regression_feature_selection(Xy1_imputed, y1)
  print("Stepwise Regression Selected Feature for Y1:", selected_feature_y1_stepwise)
  
  # Perform Stepwise Regression feature selection for Y2
  selected_feature_y2_stepwise = stepwise_regression_feature_selection(Xy2_imputed, y2)
  print("Stepwise Regression Selected Feature for Y2:", selected_feature_y2_stepwise)
