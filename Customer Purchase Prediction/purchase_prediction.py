import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Customer Purchase Prediction")

df = pd.read_csv("Customer Purchase Prediction/customer_data.csv")

df = df.ffill()

le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

le_purchase = LabelEncoder()
df["Purchase"] = le_purchase.fit_transform(df["Purchase"])

X = df.drop("Purchase", axis=1)
y = df["Purchase"]

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

metrics_df = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_lr),
            accuracy_score(y_test, y_pred_rf),
        ],
        "Precision": [
            precision_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_rf),
        ],
        "Recall": [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf)],
        "F1 Score": [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)],
    }
)

st.subheader("Model Performance")
st.dataframe(metrics_df, width="stretch")

model = rf

st.subheader("Predict New Customer")

age = st.slider("Age", 18, 60)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Income")
price = st.number_input("Product Price")
prev_purchases = st.number_input("Previous Purchases")

gender_encoded = le_gender.transform([gender])[0]

if st.button("Predict"):
    input_df = pd.DataFrame(
        [[age, gender_encoded, income, price, prev_purchases]], columns=X.columns
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Will Purchase")
    else:
        st.error("Will Not Purchase")
