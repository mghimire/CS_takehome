import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from dataimport.py import import_csv_parquet

def elastic_net_feature_selection(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Build an Elastic Net model with cross-validated alpha selection
    elastic_net_model = ElasticNetCV(cv=5)
    elastic_net_model.fit(X_train_std, y_train)

    # Display the selected features and their corresponding coefficients
    selected_features = pd.Series(elastic_net_model.coef_, index=X.columns)
    selected_features = selected_features[selected_features != 0]

    return selected_features.index.tolist()


def stepwise_regression_feature_selection(X, y):
    # Add a constant column to the features for the intercept term
    X_with_intercept = sm.add_constant(X)

    # Perform stepwise regression using statsmodels
    stepwise_model = sm.OLS(y, X_with_intercept).fit()
    
    # Display summary statistics
    print(stepwise_model.summary())

    # Get the most relevant features based on p-values
    selected_features = stepwise_model.pvalues[1:].idxmin()

    return selected_features

if __name__ == "__main__":

  filepath = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220103.csv.parquet'
  df = import_csv_parquet(filepath)
  
  # Extract dependent variables Y1 and Y2
  y1 = df['Y1']
  y2 = df['Y2']
  
  # Extract independent variables X1 to X375
X = df.drop(['time', 'Q1', 'Q2', 'Y1', 'Y2'], axis=1)
  
  # Perform Elastic Net feature selection for Y1
  selected_features_y1_elastic_net = elastic_net_feature_selection(X, y1)
  print("Elastic Net Selected Features for Y1:", selected_features_y1_elastic_net)
  
  # Perform Elastic Net feature selection for Y2
  selected_features_y2_elastic_net = elastic_net_feature_selection(X, y2)
  print("Elastic Net Selected Features for Y2:", selected_features_y2_elastic_net)
  
  # Perform Stepwise Regression feature selection for Y1
  selected_feature_y1_stepwise = stepwise_regression_feature_selection(X, y1)
  print("Stepwise Regression Selected Feature for Y1:", selected_feature_y1_stepwise)
  
  # Perform Stepwise Regression feature selection for Y2
  selected_feature_y2_stepwise = stepwise_regression_feature_selection(X, y2)
  print("Stepwise Regression Selected Feature for Y2:", selected_feature_y2_stepwise)
