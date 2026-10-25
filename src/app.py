
from pathlib import Path
from src.database import (
    initialize_database,
    save_prediction,
    get_predictions,
)
import joblib
import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import nn


# Locate saved model files
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

CHECKPOINT_PATH = MODEL_DIR / "neuroguard_nn.pt"
LEGACY_CHECKPOINT_PATH = MODEL_DIR / "neuroguard_nn.pth"
SCALER_PATH = MODEL_DIR / "neuroguard_scaler.joblib"


# Recreate the architecture used during training
class RULNetwork(nn.Module):
    def __init__(self, input_size):
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


checkpoint_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else LEGACY_CHECKPOINT_PATH
if not checkpoint_path.exists():
    raise FileNotFoundError(
        "No trained model checkpoint found. Expected one of: "
        f"{CHECKPOINT_PATH} or {LEGACY_CHECKPOINT_PATH}"
    )

# Load the saved artifacts. Keep the checkpoint flexible because the training script
# saves a dict containing the scaler object as well as the model state.
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

FEATURE_COLUMNS = checkpoint["feature_columns"]

model = RULNetwork(len(FEATURE_COLUMNS))
state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
model.load_state_dict(checkpoint[state_dict_key])
model.eval()

if "scaler" in checkpoint:
    scaler = checkpoint["scaler"]
elif SCALER_PATH.exists():
    scaler = joblib.load(SCALER_PATH)
else:
    raise FileNotFoundError(f"No scaler found at {SCALER_PATH} or in checkpoint {checkpoint_path}")

app = FastAPI(title="NeuroGuard AI API")
initialize_database()


# Define the request format
class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "project": "NeuroGuard AI",
        "status": "running",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    # Check that every expected feature is provided
    missing = set(FEATURE_COLUMNS) - set(request.features)
    extra = set(request.features) - set(FEATURE_COLUMNS)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(extra),
            },
        )

    # Arrange values in the exact training order
    values = np.array(
        [[request.features[name] for name in FEATURE_COLUMNS]],
        dtype=np.float32,
    )

    # Apply the same preprocessing used during training
    scaled_values = scaler.transform(values)

    # Generate the prediction
    with torch.no_grad():
        tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32,
        )
        prediction = model(tensor).item()

    # RUL cannot be negative in this API's output
    prediction = max(0.0, prediction)

    prediction_id = save_prediction(prediction)

    return {
        "prediction_id": prediction_id,
        "predicted_rul_cycles": round(prediction, 2),
        "unit": "operating cycles",
    }


@app.get("/predictions")
def prediction_history(limit: int = 20):
    if not 1 <= limit <= 100:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 100",
        )

    return {
        "predictions": get_predictions(limit)
    }