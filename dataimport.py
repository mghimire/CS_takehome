import os
import pandas as pd
import dask.dataframe as dd


from cleaning import *

def import_csv_parquet(file_path):
    # Read the Parquet file into a pandas DataFrame
    ddf = dd.read_parquet(file_path, engine='fastparquet')
    df = ddf.compute()

    df = removebadval(df)
    return skipbadrow(df)

def import_pct(directory_path = '..', pct=100, fromstart=True):
    # This function is prone to running into memory issues because it attemps to
    # handle all the data at once. It is only recommended to be used on a powerful
    # enough computer.

    # Get a list of all files in the directory
    all_files = os.listdir(directory_path)

    # Filter files to include only *.csv.parquet files
    csv_parquet_files = [file for file in all_files if file.endswith('.csv.parquet')]

    #Determine which files to include based on percentage required and if from start
    numfiles = int(len(csv_parquet_files)*pct/100.)
    if fromstart:
        csv_parquet_files = csv_parquet_files[:numfiles]
    else:
        csv_parquet_files = csv_parquet_files[numfiles:]

    # Check if there are any files to process
    if not csv_parquet_files:
        print("No *.csv.parquet files found in the specified directory.")
        return None

    # Initialize an empty Dask DataFrame
    merged_df = dd.from_pandas(pd.DataFrame(), npartitions=4)  # Adjust the number of partitions based on your system's cores

    # Loop through each file and merge its data into the main DataFrame
    for file in csv_parquet_files:
        file_path = os.path.join(directory_path, file)
        df = import_csv_parquet(file_path)
        merged_df = dd.concat([merged_df, dd.from_pandas(df, npartitions=4)], ignore_index=True)

    return merged_df.compute()

if __name__ == "__main__":
  # Specify the path to the *.csv.parquet files
  parquet_file_path = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220104.csv.parquet'
  
  # Import the data into a pandas dataframe array
  df = import_csv_parquet(parquet_file_path)
  
  # Display the pd
  print(df)
