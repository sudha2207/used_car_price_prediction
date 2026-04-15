import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# LOAD & TRAIN MODEL
df = pd.read_csv("data/cardekho_dataset.csv")

df.drop(["Unnamed: 0", "car_name", "model"], axis=1, inplace=True)

df["mileage"] = df["mileage"].astype(str).str.replace(" kmpl", "")
df["mileage"] = pd.to_numeric(df["mileage"], errors='coerce')

df["engine"] = df["engine"].astype(str).str.replace(" CC", "")
df["engine"] = pd.to_numeric(df["engine"], errors='coerce')

df["max_power"] = df["max_power"].astype(str).str.replace(" bhp", "")
df["max_power"] = pd.to_numeric(df["max_power"], errors='coerce')

df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

model = RandomForestRegressor()
model.fit(X, y)

# UI STARTS HERE
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")

st.title("🚗 Used Car Price Predictor")
st.write("Enter the details below to estimate the selling price of a used car.")

st.subheader("Car Details")

vehicle_age = st.select_slider(
    "Vehicle Age (Years)",
    options=list(range(0, 21))
)

km_driven = st.select_slider(
    "Kilometers Driven",
    options=[0, 5000, 10000, 20000, 30000, 50000, 70000, 100000, 150000, 200000]
)

mileage = st.slider(
    "Mileage (kmpl)",
    min_value=5.0,
    max_value=40.0,
    value=18.0
)

engine = st.select_slider(
    "Engine (CC)",
    options=[800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 4000, 5000]
)

max_power = st.select_slider(
    "Max Power (BHP)",
    options=[40, 60, 80, 100, 120, 150, 200, 300, 400, 500]
)

seats = st.radio(
    "Number of Seats",
    [2, 4, 5, 6, 7, 8]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "LPG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Individual", "Dealer"]
)

transmission_type = st.selectbox(
    "Transmission Type",
    ["Manual", "Automatic"]
)

# PREDICTION LOGIC
if st.button("Predict Price"):

    # create input dataframe with same columns
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0

    # fill numeric values
    input_data["vehicle_age"] = vehicle_age
    input_data["km_driven"] = km_driven
    input_data["mileage"] = mileage
    input_data["engine"] = engine
    input_data["max_power"] = max_power
    input_data["seats"] = seats

    # handle categorical values
    if f"fuel_type_{fuel_type}" in input_data.columns:
        input_data[f"fuel_type_{fuel_type}"] = 1

    if f"seller_type_{seller_type}" in input_data.columns:
        input_data[f"seller_type_{seller_type}"] = 1

    if f"transmission_type_{transmission_type}" in input_data.columns:
        input_data[f"transmission_type_{transmission_type}"] = 1

    # prediction
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")