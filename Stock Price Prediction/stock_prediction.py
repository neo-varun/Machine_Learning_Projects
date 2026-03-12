import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Stock Price Prediction/stock_data.csv")

data["Date"] = pd.to_datetime(data["Date"])

data.set_index("Date", inplace=True)

data.ffill(inplace=True)

X = data[["Open", "High", "Low", "Volume"]]

y = data["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE: ", rmse)


def predict_price(days):

    stock_data = data.tail(days)

    X_input = stock_data[["Open", "High", "Low", "Volume"]]

    prediction = model.predict(X_input)

    return prediction[0]


while True:

    days = int(input("Enter the number of days to use for price prediction: "))

    price = predict_price(days)

    print("Predicted price for next day:", round(price,2))
