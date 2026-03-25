import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("House Price Prediction/house_data.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)

le = LabelEncoder()
df["Location"] = le.fit_transform(df["Location"])

X = df.drop("Price", axis=1)
y = df["Price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

st.title("House Price Prediction")

model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

area = st.number_input("Area", min_value=500, max_value=5000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10)
location = st.selectbox("Location", le.classes_)
year = st.number_input("Year Built", min_value=1950, max_value=2025)

loc_encoded = le.transform([location])[0]

if model_choice == "Linear Regression":
    model = lr
    mae = lr_mae
    mse = lr_mse
    r2 = lr_r2
else:
    model = rf
    mae = rf_mae
    mse = rf_mse
    r2 = rf_r2

if st.button("Predict"):
    input_df = pd.DataFrame(
        [[area, bedrooms, loc_encoded, year]],
        columns=["Area", "Bedrooms", "Location", "YearBuilt"],
    )
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.write(int(prediction[0]))

st.subheader("Evaluation")

st.write("MAE:", mae)
st.write("MSE:", mse)
st.write("R2:", r2)
