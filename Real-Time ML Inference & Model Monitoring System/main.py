from fastapi import FastAPI
import pandas as pd
import joblib
from datetime import datetime
import os

app = FastAPI()

MODEL_PATH = "sales_model.pkl"
LOG_PATH = "sales_log.csv"

artifact = joblib.load(MODEL_PATH)

model = artifact["linear_regression"]
feature_columns = artifact["feature_columns"]

if not os.path.exists(LOG_PATH):
    pd.DataFrame(
        columns=["timestamp", "days", "store", "promotion", "prediction"]
    ).to_csv(LOG_PATH, index=False)


def build_input(days, store, promotion):
    today = datetime.now()

    rows = []
    for i in range(1, days + 1):
        future_date = today + pd.Timedelta(days=i)

        row = {
            "Day": future_date.day,
            "Month": future_date.month,
            "Promotions": promotion,
        }

        for col in feature_columns:
            if col.startswith("Store_"):
                row[col] = 1 if col == f"Store_{store}" else 0

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
def predict(days: int, store: str, promotion: int):
    input_df = build_input(days, store, promotion)

    preds = model.predict(input_df)

    result = preds.tolist()

    log = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(),
                "days": days,
                "store": store,
                "promotion": promotion,
                "prediction": sum(result) / len(result),
            }
        ]
    )

    log.to_csv(LOG_PATH, mode="a", header=False, index=False)

    return {
        "predictions": result,
        "average_prediction": float(sum(result) / len(result)),
    }


@app.get("/logs")
def logs():
    df = pd.read_csv(LOG_PATH)
    return df.to_dict(orient="records")
