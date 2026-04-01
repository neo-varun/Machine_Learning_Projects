import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("Sales Forecasting App")

df = pd.read_csv("Sales Forecasting/sales_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month

df = pd.get_dummies(df, columns=["Store"], drop_first=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop(["Sales", "Date"], axis=1)
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

lr_rmse = np.sqrt(lr_mse)
rf_rmse = np.sqrt(rf_mse)

st.subheader("Model Selection")
model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])

if model_choice == "Linear Regression":
    selected_model = lr
    mse = lr_mse
    rmse = lr_rmse
else:
    selected_model = rf
    mse = rf_mse
    rmse = rf_rmse

st.subheader("Model Performance")
st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")

st.subheader("Predict Future Sales")

future_days = st.slider("Days to Predict", 1, 30, 5)
store_input = st.selectbox("Select Store", ["North", "South", "East", "West"])
promo_input = st.selectbox("Promotion", [0, 1])

last_date = df["Date"].max()
future_data = []

for i in range(1, future_days + 1):
    new_date = last_date + pd.Timedelta(days=i)
    row = {"Day": new_date.day, "Month": new_date.month, "Promotions": promo_input}

    for col in X.columns:
        if col.startswith("Store_"):
            row[col] = 1 if col == f"Store_{store_input}" else 0

    future_data.append(row)

future_df = pd.DataFrame(future_data)
future_df = future_df.reindex(columns=X.columns, fill_value=0)

predictions = selected_model.predict(future_df)

st.write("Future Predictions:")
st.line_chart(predictions)
