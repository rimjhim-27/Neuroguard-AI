
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Make results more reproducible
torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cmapss" / "train_FD001.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "neuroguard_nn.pt"

FEATURE_COLUMNS = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=[
            "unit_id",
            "cycle",
            "setting_1",
            "setting_2",
            "setting_3",
        ] + [f"sensor_{i}" for i in range(1, 22)],
    )

    df["max_cycle"] = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df


class RULNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def main() -> None:
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df["RUL"]
    groups = df["unit_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training engines:", groups.iloc[train_idx].nunique())
    print("Validation engines:", groups.iloc[val_idx].nunique())

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    print("Training tensor shape:", X_train_tensor.shape)
    print("Target tensor shape:", y_train_tensor.shape)

    model = RULNetwork(input_size=len(FEATURE_COLUMNS))
    print(model)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)

        average_loss = total_loss / len(train_dataset)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Training MSE: {average_loss:.2f}")

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor).numpy()

    mae = mean_absolute_error(y_val.to_numpy(), val_predictions)
    rmse = np.sqrt(mean_squared_error(y_val.to_numpy(), val_predictions))

    print("\nDeep Learning Results")
    print(f"MAE: {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "scaler": scaler,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()


import joblib

# Create the model directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save the trained neural network
torch.save(
    {
        "state_dict": model.state_dict(),
        "feature_columns": feature_columns,
    },
    MODEL_DIR / "neuroguard_nn.pth",
)

# Save the fitted scaler separately
joblib.dump(
    scaler,
    MODEL_DIR / "neuroguard_scaler.joblib",
)

print("Neural network and scaler saved successfully!")
