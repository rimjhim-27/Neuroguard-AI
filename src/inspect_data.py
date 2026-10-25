
from pathlib import Path
import pandas as pd

# Locate the training dataset
DATA_PATH = Path("data/cmapss/train_FD001.txt")

# Check whether the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Define the column names
columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Read the dataset
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns,
)

# Explore the data
print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nNumber of engines:")
print(df["unit_id"].nunique())

print("\nMaximum cycle number:")
print(df["cycle"].max())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())
