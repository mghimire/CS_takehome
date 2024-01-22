import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

from dataimport import import_csv_parquet
from models import FeedforwardModel
from cleaning import extract_both

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 375

# Move the model to GPU if available
feedforward_model = FeedforwardModel(input_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(feedforward_model.parameters(), lr=0.001)

model_weights_save_path = "./model_weights.pth"

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

epochs = 10

for epoch in range(epochs):
  for file in csv_parquet_files:
    file_path = os.path.join(directory_path, file)
    
    print(f"\nTraining on {file}:")
    
    df = import_csv_parquet(file_path)
    
    y, X = extract_both(df)

    # Standardize the features for the Feedforward model
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert data to PyTorch tensors and move to GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    feedforward_model.train_feedforward_model(train_loader, criterion, optimizer)
  
  # Save the model weights after each epoch
  torch.save(feedforward_model.state_dict(), model_weights_save_path)

  
print("Training complete.")