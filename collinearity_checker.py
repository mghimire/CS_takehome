import seaborn as sns
import matplotlib.pyplot as plt

from dataimport import *

# Assuming X_train is your training data (selected features only)
filepath = 'X:\Documents\qr_takehome\QR_TAKEHOME_20220215.csv.parquet'
df = import_csv_parquet(filepath)

X_train = df[(['X329', 'X8', 'X351', 'X372', 'X370', 'X320', 'X373', 'X353', 'X328', 'X68'])]

# Calculate the correlation matrix using training data
correlation_matrix = X_train.corr()

# Set a threshold for collinearity
collinearity_threshold = 0.7

# Identify highly correlated features
highly_correlated_pairs = [(feature1, feature2) for feature1 in X_train.columns
                           for feature2 in X_train.columns
                           if feature1 < feature2 and abs(correlation_matrix.loc[feature1, feature2]) > collinearity_threshold]

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Selected Features')
plt.show()

# Print highly correlated feature pairs
print("Highly Correlated Feature Pairs:")
for pair in highly_correlated_pairs:
    print(pair)