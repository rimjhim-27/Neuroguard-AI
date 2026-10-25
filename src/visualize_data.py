
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/cmapss/train_FD001.txt")

columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns,
)

# Select engine number 1
engine = df[df["unit_id"] == 1]

# Plot sensor 2 across operating cycles
plt.plot(engine["cycle"], engine["sensor_2"])

plt.xlabel("Operating cycle")
plt.ylabel("Sensor 2 reading")
plt.title("Engine 1: Sensor 2 over time")

plt.tight_layout()
plt.show()
