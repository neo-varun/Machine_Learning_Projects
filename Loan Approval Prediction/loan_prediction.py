import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Loan Approval Prediction/loan_data.csv")

df.fillna(df.median(numeric_only=True), inplace=True)

le_emp = LabelEncoder()
le_status = LabelEncoder()

df["EmploymentStatus"] = le_emp.fit_transform(df["EmploymentStatus"])
df["LoanStatus"] = le_status.fit_transform(df["LoanStatus"])

X = df.drop("LoanStatus", axis=1)
y = df["LoanStatus"]

scaler = StandardScaler()
X[["ApplicantIncome", "LoanAmount"]] = scaler.fit_transform(
    X[["ApplicantIncome", "LoanAmount"]]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

models = {
    1: ("Logistic Regression", lr),
    2: ("Decision Tree", dt),
    3: ("Random Forest", rf),
}

while True:
    print("\nSelect Model:")
    print("1 - Logistic Regression")
    print("2 - Decision Tree")
    print("3 - Random Forest")
    print("0 - Exit")

    choice = int(input("Enter choice: "))

    if choice == 0:
        break

    if choice not in models:
        print("Invalid choice")
        continue

    model_name, model = models[choice]

    applicant_income = float(input("Enter Applicant Income: "))
    loan_amount = float(input("Enter Loan Amount: "))
    credit_history = int(input("Enter Credit History (1/0): "))
    employment_status = input(
        "Enter Employment Status (Salaried/SelfEmployed/Unemployed): "
    )

    emp = le_emp.transform([employment_status])[0]

    data = pd.DataFrame(
        [[applicant_income, loan_amount, credit_history, emp]],
        columns=["ApplicantIncome", "LoanAmount", "CreditHistory", "EmploymentStatus"],
    )

    data[["ApplicantIncome", "LoanAmount"]] = scaler.transform(
        data[["ApplicantIncome", "LoanAmount"]]
    )

    pred = model.predict(data)[0]

    result = "Approved" if pred == 1 else "Rejected"

    print(f"Model Used: {model_name}")
    print(f"Loan Status: {result}")
