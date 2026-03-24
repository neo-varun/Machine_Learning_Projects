import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Employee Attrition Prediction/employee_attrition.csv")

df = df.dropna()

le_role = LabelEncoder()
le_attr = LabelEncoder()

df["JobRole"] = le_role.fit_transform(df["JobRole"])
df["Attrition"] = le_attr.fit_transform(df["Attrition"])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LogisticRegression()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)


def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    ]


metrics_lr = get_metrics(y_test, y_pred_lr)
metrics_rf = get_metrics(y_test, y_pred_rf)

evaluation_df = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [metrics_lr[0], metrics_rf[0]],
        "Precision": [metrics_lr[1], metrics_rf[1]],
        "Recall": [metrics_lr[2], metrics_rf[2]],
        "F1 Score": [metrics_lr[3], metrics_rf[3]],
    }
)

st.title("Employee Attrition Prediction System")

st.subheader("Model Evaluation Table")
st.table(evaluation_df)

st.subheader("Select Model for Prediction")

model_choice = st.radio("Choose Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    selected_model = lr
else:
    selected_model = rf

st.subheader("Predict Employee Attrition")

age = st.number_input("Age", 18, 60)
salary = st.number_input("Salary", 20000, 100000)
job_role = st.selectbox("Job Role", le_role.classes_)
experience = st.number_input("Work Experience", 0, 40)
satisfaction = st.slider("Job Satisfaction", 1, 5)

if st.button("Predict"):
    role_encoded = le_role.transform([job_role])[0]
    input_df = pd.DataFrame(
        [[age, salary, role_encoded, experience, satisfaction]],
        columns=["Age", "Salary", "JobRole", "WorkExperience", "JobSatisfaction"],
    )
    input_scaled = scaler.transform(input_df)
    prediction = selected_model.predict(input_scaled)[0]
    result = "Likely to Leave" if prediction == 1 else "Likely to Stay"
    st.success(result)
