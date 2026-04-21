import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("Federated Learning Simulation")

data = pd.read_csv("Federated Learning System/data.csv")

X = data.drop(columns=["id", "label"]).values
y = data["label"].values

num_clients = st.slider("Number of Clients", 2, 10, 3)
rounds = st.slider("Communication Rounds", 1, 10, 5)

shuffled = data.sample(frac=1).reset_index(drop=True)
client_data = [shuffled.iloc[i::num_clients] for i in range(num_clients)]


def train_local_model(df):
    X_local = df.drop(columns=["id", "label"]).values
    y_local = df["label"].values
    model = LogisticRegression(max_iter=200)
    model.fit(X_local, y_local)
    return model, len(df)


def federated_avg(models, sizes):
    total = sum(sizes)
    coef = sum(m.coef_ * (sizes[i] / total) for i, m in enumerate(models))
    intercept = sum(m.intercept_ * (sizes[i] / total) for i, m in enumerate(models))
    global_model = LogisticRegression()
    global_model.coef_ = coef
    global_model.intercept_ = intercept
    global_model.classes_ = models[0].classes_
    return global_model


global_model = None
fed_accuracies = []

for r in range(rounds):
    local_models = []
    sizes = []
    for client in client_data:
        model, size = train_local_model(client)
        local_models.append(model)
        sizes.append(size)
    global_model = federated_avg(local_models, sizes)

    X_temp = shuffled.drop(columns=["id", "label"]).values
    y_temp = shuffled["label"].values
    preds = global_model.predict(X_temp)
    acc = accuracy_score(y_temp, preds)
    fed_accuracies.append(acc)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

central_model = LogisticRegression(max_iter=200)
central_model.fit(X_train, y_train)

fed_preds = global_model.predict(X_test)
central_preds = central_model.predict(X_test)

fed_acc = accuracy_score(y_test, fed_preds)
central_acc = accuracy_score(y_test, central_preds)

st.subheader("Prediction")

f1 = st.number_input("Feature 1", value=5.0)
f2 = st.number_input("Feature 2", value=3.0)
f3 = st.number_input("Feature 3", value=2.0)

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3]])
    prediction = global_model.predict(input_data)
    st.write("Predicted Class:", int(prediction[0]))

st.subheader("Final Results")
st.write("Federated Accuracy:", fed_acc)
st.write("Centralized Accuracy:", central_acc)

st.subheader("Federated Accuracy per Round")
st.line_chart(fed_accuracies)
