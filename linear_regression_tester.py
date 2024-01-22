import os
import joblib
from dataimport import import_csv_parquet
from cleaning import extract
from models import RMSE

# Load the saved linear models
model_y1_filename = "linear_model_y1_elnet.joblib"
model_y2_filename = "linear_model_y2_elnet.joblib"

linear_model_y1 = joblib.load(model_y1_filename)
linear_model_y2 = joblib.load(model_y2_filename)

#Set the selected features:
features1 = ['X304', 'X239', 'X118', 'X25', 'X119', 'X86', 'X87', 'X26', 'X116', 'X84']
features2 = ['X304', 'X322', 'X25', 'X84', 'X85', 'X325', 'X117', 'X116', 'X239', 'X118']

cum_err_1 = 0
cum_err_2 = 0

# Specify the directory path for test data
test_directory_path = ".."
pct_data_to_process = 70
fromstart = False

# Filter files to include only test data files
test_files = [file for file in os.listdir(test_directory_path) if file.endswith('.csv.parquet')]

numfiles = int(len(test_files)*pct_data_to_process/100.)
if fromstart:
    test_files = test_files[:numfiles]
else:
    test_files = test_files[numfiles:]


# Check if there are any test files
if not test_files:
    raise Exception("No test data files found in the specified directory.")

for test_file in test_files:
    test_file_path = os.path.join(test_directory_path, test_file)

    print(f"\nTesting on {test_file}:")

    # Import test data
    test_df = import_csv_parquet(test_file_path)

    # Extract test data
    test_y1, test_Xy1, _ = extract(test_df, 1, features1)
    test_y2, test_Xy2, _ = extract(test_df, 2, features2)

    # Preprocess test data
    test_Xy1 = test_Xy1.interpolate(method='pad', axis=0).ffill().bfill()
    test_Xy2 = test_Xy2.interpolate(method='pad', axis=0).ffill().bfill()

    # Test the linear models
    predictions_y1 = linear_model_y1.predict(test_Xy1)
    predictions_y2 = linear_model_y2.predict(test_Xy2)

    # Calculate average RMSE
    RMSE_y1 = RMSE(test_y1, predictions_y1)
    RMSE_y2 = RMSE(test_y2, predictions_y2)

    cum_err_1 += RMSE_y1
    cum_err_2 += RMSE_y2

    print("Root Mean Squared Error (Y1):", RMSE_y1)
    print("Root Mean Squared Error (Y2):", RMSE_y2)

print("Average Root Mean Squared Error across all test data (Y1):", cum_err_1/len(test_files))
print("Average Root Mean Squared Error across all test data (Y2):", cum_err_2/len(test_files))

