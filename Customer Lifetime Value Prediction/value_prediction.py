import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.title("Customer Lifetime Value Prediction")

data = pd.read_csv("Customer Lifetime Value Prediction/customer_data.csv")

data["LastPurchaseDate"] = pd.to_datetime(data["LastPurchaseDate"])
data["DaysSinceLastPurchase"] = (
    pd.Timestamp.today() - data["LastPurchaseDate"]
).dt.days

X = data[["PurchaseFrequency", "AverageOrderValue", "DaysSinceLastPurchase"]]
y = data["CLV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.subheader("Choose Model")
model_choice = st.selectbox("Select a model", ["Linear Regression", "Random Forest"])

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor()

model.fit(X_train, y_train)

preds = model.predict(X_test)

st.subheader("Model Evaluation")

st.write("MAE:", mean_absolute_error(y_test, preds))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

st.subheader("Predict Customer CLV")

pf = st.number_input("Purchase Frequency", min_value=1, value=5)
aov = st.number_input("Average Order Value", min_value=10, value=150)
days = st.number_input("Days Since Last Purchase", min_value=1, value=30)

input_data = pd.DataFrame(
    [[pf, aov, days]],
    columns=["PurchaseFrequency", "AverageOrderValue", "DaysSinceLastPurchase"],
)

result = model.predict(input_data)

st.write(f"Predicted CLV ({model_choice}):", round(result[0], 2))
