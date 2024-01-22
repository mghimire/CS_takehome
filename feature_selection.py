import pandas as pd
import numpy as np
import os

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold

from dataimport import import_csv_parquet
from cleaning import cleanQs

class Feature_Selection:
    def __init__(self, model_type='lasso', alpha=1.0, l1_ratio=0.5, tol=0.001, max_features=10, drop_ratio_threshold=0.1):
        """
        Feature Selection Tracker Class.

        Parameters:
            - model_type (str): Type of the linear model ('lasso', 'ridge', or 'elasticnet').
            - alpha (float): Regularization strength.
            - l1_ratio (float): Elastic Net mixing parameter (only for 'elasticnet').
            - tol (float): Tolerance for stopping criterion.
            - max_features (int): Maximum number of features to select.
            - drop_ratio_threshold (float): Threshold ratio for dropping features based on coefficient drop.

        Attributes:
            - model_type (str): Type of the linear model.
            - alpha (float): Regularization strength.
            - l1_ratio (float): Elastic Net mixing parameter.
            - tol (float): Tolerance for stopping criterion.
            - max_features (int): Maximum number of features to select.
            - drop_ratio_threshold (float): Threshold ratio for dropping features based on coefficient drop.
            - model (object): Linear model (Lasso, Ridge, or Elastic Net).
            - sfm (object): Feature selector using SelectFromModel.
            - coefficients (list of arrays): Coefficients of the model at each update.
            - feature_names (list): Column names of the features.
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_features = max_features
        self.drop_ratio_threshold = drop_ratio_threshold
        self.model = self._initialize_model()
        self.sfm = SelectFromModel(self.model)
        self.coefficients = []
        self.scaler = StandardScaler()

    def _initialize_model(self):
        """
        Internal method to initialize the linear model based on the specified type.

        Returns:
            - object: Initialized linear model.
        """
        if self.model_type == 'lasso':
            return Lasso(alpha=self.alpha, tol=self.tol)
        elif self.model_type == 'ridge':
            return Ridge(alpha=self.alpha, tol=self.tol)
        elif self.model_type == 'elasticnet':
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, tol=self.tol)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def update_feature_selection(self, X, y):
        """
        Update feature selection based on a chunk of data.

        Parameters:
            - X (array-like): Feature matrix.
            - y (array-like): Target variable.
        """
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.coefficients.append(self.model.coef_)

    def get_selected_features(self):
        if not self.coefficients:
            raise ValueError("No coefficients available. Please run update_feature_selection first.")

        # Stack coefficients across updates
        all_coefficients = np.vstack(self.coefficients)

        # Find the mean of each coefficient across updates
        mean_coefficients = np.mean(all_coefficients, axis=0)

        # Identify the top features based on highest mean coefficients
        top_feature_indices = np.argsort(np.abs(mean_coefficients))[::-1][:self.max_features]

        # Initialize the dictionary of selected features
        selected_features_dict = {}

        # Iterate through the top feature indices
        for idx in top_feature_indices:
            # Check if the current coefficient is above the drop threshold
            if np.abs(mean_coefficients[idx]) >= self.drop_ratio_threshold * np.abs(mean_coefficients[top_feature_indices[0]]):
                # Get the name of the selected feature
                feature_name = f'X{idx + 1}'
                # Add the feature name and its mean coefficient to the dictionary
                selected_features_dict[feature_name] = mean_coefficients[idx]

        return selected_features_dict

if __name__ == "__main__":
  # Configure parameters for script
  directory_path = ".."
  pct_data_to_process = 70
  fromstart = True

  # Initialize all selector modules
  lasso_selector_y1 = Feature_Selection(model_type='lasso')
  lasso_selector_y2 = Feature_Selection(model_type='lasso')
  ridge_selector_y1 = Feature_Selection(model_type='ridge')
  ridge_selector_y2 = Feature_Selection(model_type='ridge')
  elnet_selector_y1 = Feature_Selection(model_type='elasticnet')
  elnet_selector_y2 = Feature_Selection(model_type='elasticnet')
  
  # Get a list of all files in the directory
  all_files = os.listdir(directory_path)

  # Filter files to include only *.csv.parquet files
  csv_parquet_files = [file for file in all_files if file.endswith('.csv.parquet')]
  
  numfiles = int(len(csv_parquet_files)*pct_data_to_process/100.)
  if fromstart:
    csv_parquet_files = csv_parquet_files[:numfiles]
  else:
    csv_parquet_files = csv_parquet_files[numfiles:]

    # Check if there are any files to process
  if not csv_parquet_files:
    raise Exception("No *.csv.parquet files found in the specified directory.")

  #Debugging script
  #csv_parquet_files = ["QR_TAKEHOME_20220107.csv.parquet"]

  for file in csv_parquet_files:

    file_path = os.path.join(directory_path, file)
    print(file_path)
    df = import_csv_parquet(file_path)
    
    # Separate df into dfy1 and dfy2 based on good 'Q1' and 'Q2' values
    dfy1 = cleanQs(df, 1)
    dfy2 = cleanQs(df, 2)

    # Extract dependent variables Y1 and Y2
    y1 = dfy1['Y1']
    y2 = dfy2['Y2']
    
    # Extract independent variables X1 to X375
    Xy1 = dfy1.drop(['time', 'sym', 'exch', 'Q1', 'Q2', 'Y1', 'Y2'], axis=1)
    Xy2 = dfy2.drop(['time', 'sym', 'exch', 'Q1', 'Q2', 'Y1', 'Y2'], axis=1)

    # Handle NaN values by linear imputation (use spline interpolation instead)
    Xy1 = Xy1.interpolate(method='pad', axis=0).ffill().bfill()
    Xy2 = Xy2.interpolate(method='pad', axis=0).ffill().bfill()

    lasso_selector_y1.update_feature_selection(Xy1, y1)
    lasso_selector_y2.update_feature_selection(Xy2, y2)
    ridge_selector_y1.update_feature_selection(Xy1, y1)
    ridge_selector_y2.update_feature_selection(Xy2, y2)
    elnet_selector_y1.update_feature_selection(Xy1, y1)
    elnet_selector_y2.update_feature_selection(Xy2, y2)

  # Print selected feature keys for Y1
  selected_features_y1_lasso = lasso_selector_y1.get_selected_features()
  print("Lasso Selected Features for Y1:", list(selected_features_y1_lasso.keys()))

  selected_features_y1_ridge = ridge_selector_y1.get_selected_features()
  print("Ridge Selected Features for Y1:", list(selected_features_y1_ridge.keys()))

  selected_features_y1_elnet = elnet_selector_y1.get_selected_features()
  print("Elastic Net Selected Features for Y1:", list(selected_features_y1_elnet.keys()))

  # Print selected feature keys for Y2
  selected_features_y2_lasso = lasso_selector_y2.get_selected_features()
  print("Lasso Selected Features for Y2:", list(selected_features_y2_lasso.keys()))

  selected_features_y2_ridge = ridge_selector_y2.get_selected_features()
  print("Ridge Selected Features for Y2:", list(selected_features_y2_ridge.keys()))

  selected_features_y2_elnet = elnet_selector_y2.get_selected_features()
  print("Elastic Net Selected Features for Y2:", list(selected_features_y2_elnet.keys()))