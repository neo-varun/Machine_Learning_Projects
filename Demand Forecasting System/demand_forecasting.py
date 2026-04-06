import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Demand Forecasting App")

df = pd.read_csv("Demand Forecasting System/demand_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month

X = df[["Product_ID", "Price", "Promotion", "Day", "Month"]]
y = df["Sales_Quantity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])

st.subheader("Model Evaluation")

if model_choice == "Linear Regression":
    pred = lr.predict(X_test)
    st.write("Model: Linear Regression")
    st.write("MAE:", mean_absolute_error(y_test, pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
else:
    pred = rf.predict(X_test)
    st.write("Model: Random Forest")
    st.write("MAE:", mean_absolute_error(y_test, pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

st.subheader("Predict Future Demand")

product_id = st.number_input("Product ID", min_value=1000, max_value=1010, value=1001)
price = st.slider("Price", 10.0, 100.0, 50.0)
promotion = st.selectbox("Promotion", [0, 1])
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)

input_data = pd.DataFrame(
    {
        "Product_ID": [product_id],
        "Price": [price],
        "Promotion": [promotion],
        "Day": [day],
        "Month": [month],
    }
)

if st.button("Predict"):
    if model_choice == "Linear Regression":
        result = lr.predict(input_data)[0]
    else:
        result = rf.predict(input_data)[0]

    st.success(f"Predicted Demand: {result:.2f}")
