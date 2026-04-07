import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Credit Risk Scoring System/credit_data.csv")

le_ch = LabelEncoder()
le_emp = LabelEncoder()
le_def = LabelEncoder()

df["CreditHistory"] = le_ch.fit_transform(df["CreditHistory"])
df["EmploymentStatus"] = le_emp.fit_transform(df["EmploymentStatus"])
df["Default"] = le_def.fit_transform(df["Default"])

X = df.drop("Default", axis=1)
y = df["Default"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LogisticRegression()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

st.title("Credit Risk Scoring System")

model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = lr
    y_pred = model.predict(X_test)
else:
    model = rf
    y_pred = model.predict(X_test)

st.subheader("Model Evaluation")

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))

st.subheader("Predict Credit Risk")

income = st.number_input("Income", value=50000)
loan = st.number_input("Loan Amount", value=20000)
credit = st.selectbox("Credit History", ["Good", "Average", "Poor"])
employment = st.selectbox(
    "Employment Status", ["Employed", "Self-Employed", "Unemployed"]
)

if st.button("Predict"):
    credit_val = le_ch.transform([credit])[0]
    emp_val = le_emp.transform([employment])[0]
    input_df = pd.DataFrame([[income, loan, credit_val, emp_val]], columns=X.columns)
    input_data = scaler.transform(input_df)
    pred = model.predict(input_data)[0]

    if pred == 0:
        risk = "Low Risk"
    elif pred == 1:
        risk = "High Risk"
    else:
        risk = "Medium Risk"

    st.write("Prediction:", risk)
