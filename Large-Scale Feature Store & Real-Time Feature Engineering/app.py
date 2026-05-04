import pandas as pd
import numpy as np
import random
import time
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

n = 100
data = pd.DataFrame(
    {
        "user_id": np.random.randint(1, 6, n),
        "amount": np.random.randint(10, 1000, n),
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h"),
        "is_fraud": np.random.randint(0, 2, n),
    }
)

data = data.sort_values(by=["user_id", "timestamp"])

data["avg_txn_5"] = data.groupby("user_id")["amount"].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

data["txn_count"] = data.groupby("user_id").cumcount() + 1

offline_features = data.copy()

X = offline_features[["avg_txn_5", "txn_count"]]
y = offline_features["is_fraud"]

model = RandomForestClassifier()
model.fit(X, y)

online_store = {}

for user_id in data["user_id"].unique():
    user_data = data[data["user_id"] == user_id]
    last_row = user_data.iloc[-1]

    online_store[user_id] = {
        "avg_txn_5": last_row["avg_txn_5"],
        "txn_count": last_row["txn_count"],
        "last_amount": last_row["amount"],
        "last_timestamp": last_row["timestamp"],
    }


def compute_realtime_features(event):
    user_id = event["user_id"]
    amount = event["amount"]
    current_time = event["timestamp"]

    user_state = online_store.get(user_id, None)

    if user_state is None:
        return {
            "avg_txn_5": 0,
            "txn_count": 1,
            "amount_deviation": 0,
            "time_since_last_txn": 0,
        }

    amount_deviation = amount - user_state["avg_txn_5"]
    time_since_last_txn = (current_time - user_state["last_timestamp"]).total_seconds()

    return {
        "avg_txn_5": user_state["avg_txn_5"],
        "txn_count": user_state["txn_count"],
        "amount_deviation": amount_deviation,
        "time_since_last_txn": time_since_last_txn,
    }


def update_online_store(event):
    user_id = event["user_id"]
    amount = event["amount"]
    timestamp = event["timestamp"]

    state = online_store.get(
        user_id,
        {"avg_txn_5": 0, "txn_count": 0, "last_amount": 0, "last_timestamp": timestamp},
    )

    new_count = state["txn_count"] + 1
    new_avg = ((state["avg_txn_5"] * state["txn_count"]) + amount) / new_count

    online_store[user_id] = {
        "avg_txn_5": new_avg,
        "txn_count": new_count,
        "last_amount": amount,
        "last_timestamp": timestamp,
    }


def predict(event):
    features = compute_realtime_features(event)

    model_input = pd.DataFrame(
        [{"avg_txn_5": features["avg_txn_5"], "txn_count": features["txn_count"]}]
    )

    prediction = model.predict(model_input)[0]

    update_online_store(event)

    return prediction, features


try:
    while True:
        event = {
            "user_id": random.randint(1, 5),
            "amount": random.randint(10, 1000),
            "timestamp": pd.Timestamp.now(),
        }

        pred, feats = predict(event)

        print(f"Event: {event}")
        print(f"Features: {feats}")
        print(f"Prediction: {pred}")
        print("-" * 50)

        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped")
