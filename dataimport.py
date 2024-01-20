import os
import pandas as pd
from cleaning import *

def import_csv_parquet(file_path):
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(file_path, engine='pyarrow')
    df = removebadval(df)
    return skipbadrow(df)

def import_all(directory_path = '..'):
   # Get a list of all files in the directory
    all_files = os.listdir(directory_path)

    # Filter files to include only *.csv.parquet files
    csv_parquet_files = [file for file in all_files if file.endswith('.csv.parquet')]

    # Check if there are any files to process
    if not csv_parquet_files:
        print("No *.csv.parquet files found in the specified directory.")
        return None

    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Loop through each file and merge its data into the main DataFrame
    for file in csv_parquet_files:
        file_path = os.path.join(directory_path, file)
        df = import_csv_parquet(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df

if __name__ == "__main__":
  # Specify the path to the *.csv.parquet files
  parquet_file_path = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220104.csv.parquet'
  
  # Import the data into a pandas dataframe array
  df = import_csv_parquet(parquet_file_path)
  
  # Display the head
  print(df)
