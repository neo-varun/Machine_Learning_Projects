import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("Network Intrusion Detection System/network_intrusion_dataset.csv")

data.ffill(inplace=True)

encoder = LabelEncoder()

protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}

data["Protocol"] = data["Protocol"].map(protocol_map)

label_map = {"Normal": 0, "Attack": 1}
data["Label"] = data["Label"].map(label_map)

scaler = StandardScaler()

numerical_cols = ["Duration", "Bytes_Sent", "Login_Attempts"]

data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

X = data.drop("Label", axis=1)
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


def evaluate(y_test, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


evaluate(y_test, lr_pred, "Logistic Regression")
evaluate(y_test, rf_pred, "Random Forest")


def predict_intrsuion():
    choice = int(input("\nChoose Model (1-Logistic Regression,2-Random Forest): "))
    protocol = input("Enter Protocol (TCP/UDP/ICMP): ").upper()
    duration = float(input("Enter Duration: "))
    bytes_sent = float(input("Enter Bytes Sent: "))
    login_attempts = int(input("Enter Login Attempts: "))

    protocol_encoded = protocol_map[protocol]

    input_data = scaler.transform([[duration, bytes_sent, login_attempts]])

    final_input = np.array([[protocol_encoded, *input_data[0]]])

    if choice == 1:
        model = lr_model
    else:
        model = rf_model

    prediction = model.predict(final_input)

    if prediction[0] == 1:
        print("Intrusion Detected")
    else:
        print("No Intrusion Detected")


while True:
    predict_intrsuion()
