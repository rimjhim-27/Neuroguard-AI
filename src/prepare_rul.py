
from pathlib import Path
import pandas as pd

# 1. Locate the training dataset
DATA_PATH = Path("data/cmapss/train_FD001.txt")

# 2. Define column names
columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# 3. Load the dataset
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns,
)

# 4. Find the final recorded cycle for each engine
df["max_cycle"] = (
    df.groupby("unit_id")["cycle"].transform("max")
)

# 5. Calculate Remaining Useful Life
df["RUL"] = df["max_cycle"] - df["cycle"]

# 6. Inspect the results for engine 1
engine_1 = df[df["unit_id"] == 1]

print(engine_1[["unit_id", "cycle", "max_cycle", "RUL"]].head())

print("\nLast five rows for engine 1:")
print(engine_1[["unit_id", "cycle", "max_cycle", "RUL"]].tail())

print("\nRUL statistics:")
print(df["RUL"].describe())
