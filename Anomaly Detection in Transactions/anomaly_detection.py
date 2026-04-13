import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

st.title("Transaction Anomaly Detection")

data = pd.read_csv("Anomaly Detection in Transactions/transactions.csv")

data["TransactionTime"] = pd.to_datetime(data["TransactionTime"])
data["Hour"] = data["TransactionTime"].dt.hour

le = LabelEncoder()
data["AreaType"] = le.fit_transform(data["AreaType"])

X = data[["TransactionAmount", "Hour", "AreaType", "Frequency"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)

st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0)
hour = st.slider("Transaction Hour", 0, 23)
area = st.selectbox("Area Type", ["Rural", "Urban", "Metropolitan"])
frequency = st.slider("Transaction Frequency", 1, 10)

if st.button("Check Transaction"):
    area_encoded = le.transform([area])[0]
    input_df = pd.DataFrame(
        [[amount, hour, area_encoded, frequency]],
        columns=["TransactionAmount", "Hour", "AreaType", "Frequency"],
    )
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == -1:
        st.error("Anomaly Detected")
    else:
        st.success("Normal Transaction")
