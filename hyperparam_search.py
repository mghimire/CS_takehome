import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from dataimport import import_csv_parquet
from cleaning import cleanQs

file_path = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220103.csv.parquet'
  
# Import the data into a pandas dataframe array
df = import_csv_parquet(file_path)

# Extract y1, Xy1 and y2, Xy2 using function from cleaning
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

scaler = StandardScaler()

Xy1 = scaler.fit_transform(Xy1)
Xy2 = scaler.fit_transform(Xy2)

# parameters to be tested on GridSearchCV
params = {"alpha":np.arange(0.1, 100, 500)}

# Number of Folds and adding the random state for replication
kf=KFold(n_splits=5,shuffle=True, random_state=42)

# Initializing the Model
lasso = Lasso()

# GridSearchCV with model, params and folds.
lasso_cv1=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv1.fit(Xy1, y1)
lasso_cv2=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv2.fit(Xy2, y2)

print("Best Params for y1 {}".format(lasso_cv1.best_params_))
print("Best Params for y2 {}".format(lasso_cv2.best_params_))