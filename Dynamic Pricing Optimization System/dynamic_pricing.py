import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

df = pd.read_csv("Dynamic Pricing Optimization System/pricing_dataset.csv")

df = df.dropna()

le = LabelEncoder()
df["Season"] = le.fit_transform(df["Season"])

X = df[["Demand", "Competitor_Price", "Season", "Current_Price"]]
y = df["Sales"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor()
gb = GradientBoostingRegressor()

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)

rf_mse = mean_squared_error(y_test, rf_pred)
gb_mse = mean_squared_error(y_test, gb_pred)

models = {
    "Random Forest": (rf, rf_mae, rf_mse),
    "Gradient Boosting": (gb, gb_mae, gb_mse),
}


def recommend_price(model, demand, competitor_price, season, current_price):
    season_enc = le.transform([season])[0]
    X_input_df = pd.DataFrame(
        [[demand, competitor_price, season_enc, current_price]],
        columns=["Demand", "Competitor_Price", "Season", "Current_Price"],
    )
    X_input = scaler.transform(X_input_df)
    sales_pred = model.predict(X_input)[0]
    optimal_price = competitor_price * (1 + (sales_pred / 1000))
    return round(optimal_price, 2)


st.title("Dynamic Pricing Optimization")

model_choice = st.selectbox("Select Model", list(models.keys()))
selected_model, selected_mae, selected_mse = models[model_choice]

demand = st.number_input("Demand", 0, 1000, 200)
competitor_price = st.number_input("Competitor Price", 0.0, 500.0, 50.0)
season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Monsoon"])
current_price = st.number_input("Current Price", 0.0, 500.0, 50.0)

if st.button("Predict Optimal Price"):
    price = recommend_price(
        selected_model, demand, competitor_price, season, current_price
    )
    st.write("Recommended Price:", price)

st.write("Selected Model MAE:", selected_mae)
st.write("Selected Model MSE:", selected_mse)
