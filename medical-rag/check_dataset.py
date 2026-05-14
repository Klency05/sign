import pandas as pd

# Load dataset
df = pd.read_csv("data/train.csv")

# Show first 5 rows
print(df.head())

# Show column names
print("\nColumns:")
print(df.columns)

# Show dataset size
print("\nDataset Shape:")
print(df.shape)
