import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Medical Diagnosis Prediction System")

st.write("Enter patient details and select model to predict disease risk")

data = pd.read_csv("Medical Diagnosis Prediction System/medical_dataset.csv")

le = LabelEncoder()
for col in ["BloodPressure", "Cholesterol", "Smoking", "Diabetes", "Obesity", "Risk"]:
    data[col] = le.fit_transform(data[col])

X = data.drop("Risk", axis=1)
y = data["Risk"]

scaler = StandardScaler()
X[["Age", "HeartRate"]] = scaler.fit_transform(X[["Age", "HeartRate"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_choice = st.selectbox(
    "Select Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"]
)

if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
else:
    model = GradientBoostingClassifier()

model.fit(X_train, y_train)

age = st.number_input("Age", min_value=1, max_value=120, value=40)

bp = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
chol = st.selectbox("Cholesterol", ["Normal", "High"])

hr = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)

smoking = st.selectbox("Smoking", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
obesity = st.selectbox("Obesity", ["No", "Yes"])

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[age, bp, chol, hr, smoking, diabetes, obesity]], columns=X.columns
    )

    for col in ["BloodPressure", "Cholesterol", "Smoking", "Diabetes", "Obesity"]:
        input_data[col] = le.fit_transform(input_data[col])

    input_data[["Age", "HeartRate"]] = scaler.transform(
        input_data[["Age", "HeartRate"]]
    )

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk (Probability: {prob:.2f})")

st.subheader("Model Evaluation")

y_pred = model.predict(X_test)

st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
