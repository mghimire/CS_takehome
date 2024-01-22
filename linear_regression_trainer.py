import os
import pandas as pd
import numpy as np
import joblib

from dataimport import import_csv_parquet
from models import LinearRegressionModel
from cleaning import extract

linear_model_y1 = LinearRegressionModel()
linear_model_y2 = LinearRegressionModel()

#Set Features
features1 = ['X304', 'X239', 'X118', 'X25', 'X119', 'X86', 'X87', 'X26', 'X116', 'X84']
features2 = ['X304', 'X322', 'X25', 'X84', 'X85', 'X325', 'X117', 'X116', 'X239', 'X118']

directory_path = ".."
pct_data_to_process = 70
fromstart = True

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
  
  print(f"\nTraining on {file}:")
  
  df = import_csv_parquet(file_path)
  
  y1, Xy1, _ = extract(df, 1, features1)
  y2, Xy2, _ = extract(df, 2, features2)

  Xy1 = Xy1.interpolate(method='pad', axis=0).ffill().bfill()
  Xy2 = Xy2.interpolate(method='pad', axis=0).ffill().bfill()

  linear_model_y1.train(Xy1, y1)
  linear_model_y2.train(Xy2, y2)

# Save the trained models to files after the loop
model_y1_filename = "linear_model_y1_elnet.joblib"
model_y2_filename = "linear_model_y2_elnet.joblib"

joblib.dump(linear_model_y1, model_y1_filename)
joblib.dump(linear_model_y2, model_y2_filename)