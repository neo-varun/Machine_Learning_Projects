import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

data = pd.read_csv("Fraud Detection System/transactions.csv")

# Fill numerical missing values with mean

numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# Fill categorical missing values with mode

categorical_cols = data.select_dtypes(include=["object", "string"]).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encode Categorical Variables

label_encoder = LabelEncoder()

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Separate Features and Label

X = data.drop("FraudLabel", axis=1)
y = data["FraudLabel"]

# Normalize Numerical Features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle Imbalanced Data

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Before SMOTE:", y.value_counts())
print("After SMOTE:", pd.Series(y_resampled).value_counts())

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train Logistic Regression

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Evaluation")

precision = precision_score(y_test, y_pred_log)
recall = recall_score(y_test, y_pred_log)
f1 = f1_score(y_test, y_pred_log)
roc_auc = roc_auc_score(y_test, y_pred_log)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)

# Train Random Forest

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Evaluation")

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)

print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("ROC-AUC Score:", roc_auc_rf)


def predict_transaction():

    print("\nSelect Model")
    print("1 - Logistic Regression")
    print("2 - Random Forest")
    model_choice = int(input("Enter model number: "))

    print("\nEnter Transaction Details")

    amount = float(input("Amount: "))
    time = int(input("Transaction Time (HHMM): "))

    print("\nMerchant Categories")
    print("1 - Cafe")
    print("2 - Clothing")
    print("3 - Electronics")
    print("4 - Food")
    print("5 - Grocery")
    print("6 - Jewelry")
    print("7 - Travel")
    merchant = int(input("Enter Merchant number: "))

    print("\nLocations")
    print("1 - Bangalore")
    print("2 - Chennai")
    print("3 - Delhi")
    print("4 - Dubai")
    print("5 - London")
    print("6 - Madurai")
    print("7 - Mumbai")
    print("8 - New York")
    print("9 - Singapore")
    location = int(input("Enter Location number: "))

    account_age = int(input("Account Age (months): "))

    print("\nTransaction Types")
    print("1 - Online")
    print("2 - POS")
    transaction_type = int(input("Enter Transaction Type number: "))

    # Create input array
    input_data = pd.DataFrame(
        [[amount, time, merchant, location, account_age, transaction_type]],
        columns=[
            "TransactionAmount",
            "TransactionTime",
            "MerchantCategory",
            "Location",
            "AccountAgeMonths",
            "TransactionType",
        ],
    )

    # Normalize input
    input_scaled = scaler.transform(input_data)

    # Model selection
    if model_choice == 1:
        prediction = log_model.predict(input_scaled)
    elif model_choice == 2:
        prediction = rf_model.predict(input_scaled)
    else:
        print("Invalid model")
        return

    if prediction[0] == 1:
        print("\nPrediction: Fraudulent Transaction")
    else:
        print("\nPrediction: Legitimate Transaction")


while True:

    predict_transaction()
