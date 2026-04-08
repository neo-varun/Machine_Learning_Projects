import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Energy Consumption Prediction")


@st.cache_data
def load_data():
    return pd.read_csv("Energy Consumption Prediction/energy_data.csv")


data = load_data()

X = data.drop("Energy_Consumption", axis=1)
y = data["Energy_Consumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

model_choice = st.selectbox("Select Model", ("Linear Regression", "Random Forest"))

if model_choice == "Linear Regression":
    preds = lr.predict(X_test_scaled)
else:
    preds = rf.predict(X_test_scaled)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

st.subheader("Model Performance")
st.write(f"Model: {model_choice}")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")

st.subheader("Predict Energy Consumption")

temp = st.slider("Temperature", 20, 40, 30)
humidity = st.slider("Humidity", 30, 80, 60)
hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day", 1, 7, 1)
usage = st.slider("Appliance Usage", 1, 5, 3)

input_data = pd.DataFrame(
    [[temp, humidity, hour, day, usage]],
    columns=["Temperature", "Humidity", "Hour", "Day", "Appliance_Usage"],
)

input_scaled = scaler.transform(input_data)

if model_choice == "Linear Regression":
    prediction = lr.predict(input_scaled)[0]
else:
    prediction = rf.predict(input_scaled)[0]

st.subheader("Prediction")
st.write(f"Predicted Energy Consumption: {prediction:.2f}")
