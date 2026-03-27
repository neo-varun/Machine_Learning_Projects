import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)

df = pd.read_csv("Multi-Model Ensemble Learning System/loan_data.csv")

st.title("Loan Approval Prediction")

df.fillna(df.mean(numeric_only=True), inplace=True)

le = LabelEncoder()
for col in ["EmploymentStatus", "MaritalStatus"]:
    df[col] = le.fit_transform(df[col])

X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LogisticRegression()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

voting = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)], voting="soft"
)

voting.fit(X_train, y_train)


def evaluate(model):
    y_pred = model.predict(X_test)
    return [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
    ]


scores_list = [evaluate(lr), evaluate(rf), evaluate(gb), evaluate(voting)]

results = pd.DataFrame(
    scores_list, columns=["Accuracy", "Precision", "Recall", "F1 Score"]
)

results.insert(
    0,
    "Model",
    ["Logistic Regression", "Random Forest", "Gradient Boosting", "Voting Ensemble"],
)

st.subheader("Model Performance")
st.dataframe(results)

st.subheader("Predict Loan Approval")

age = st.slider("Age", 20, 60)
income = st.number_input("Income")
credit = st.slider("Credit Score", 500, 800)
loan = st.number_input("Loan Amount")

employment = st.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
existing = st.slider("Existing Loans", 0, 3)

employment_map = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
marital_map = {"Single": 0, "Married": 1}

input_df = pd.DataFrame(
    [
        {
            "Age": age,
            "Income": income,
            "CreditScore": credit,
            "LoanAmount": loan,
            "EmploymentStatus": employment_map[employment],
            "ExistingLoans": existing,
            "MaritalStatus": marital_map[marital],
        }
    ]
)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    result = voting.predict(input_scaled)[0]
    if result == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")
