import pandas as pd
import numpy as np

def import_csv_parquet(file_path):
    # Read the Parquet file into a pandas DataFrame
    return pd.read_parquet(file_path, engine='pyarrow')

if __name__ == "__main__":
  # Specify the path to the *.csv.parquet files
  parquet_file_path = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220103.csv.parquet'
  
  # Import the data into a pandas dataframe array
  df = import_csv_parquet(parquet_file_path)
  
  # Display the head
  df.head()
