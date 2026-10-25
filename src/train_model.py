
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cmapss" / "train_FD001.txt"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "neuroguard_rf.joblib"

COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

FEATURE_COLUMNS = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    df["max_cycle"] = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df


def main() -> None:
    df = load_data()

    train_units, val_units = train_test_split(
        df["unit_id"].unique(),
        test_size=0.2,
        random_state=42,
    )

    train_df = df[df["unit_id"].isin(train_units)].copy()
    val_df = df[df["unit_id"].isin(val_units)].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["RUL"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["RUL"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))

    print(f"\nMAE: {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")

    results = pd.DataFrame(
        {
            "actual_RUL": y_val.to_numpy(),
            "predicted_RUL": predictions,
        }
    )
    results["error"] = results["predicted_RUL"] - results["actual_RUL"]
    results["absolute_error"] = results["error"].abs()

    print("\nLargest prediction errors:")
    print(
        results.sort_values(by="absolute_error", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

