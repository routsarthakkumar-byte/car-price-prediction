import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model and columns
model = pickle.load(open("car_price_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🚗 Car Price Prediction")

st.markdown("""
Predict the resale value of your car using Machine Learning.

Built with Streamlit and Scikit-Learn.
""")

st.sidebar.title("ℹ️ About")

st.sidebar.info("""
This application predicts used car prices
using a Machine Learning model trained
on historical car sales data.
""")

st.sidebar.success("Model Accuracy: 90% R²")

# Inputs

col1, col2 = st.columns(2)

with col1:
    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=50000
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Diesel", "Electric", "LPG", "Petrol"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

with col2:
    car_age = st.number_input(
        "Car Age",
        min_value=0,
        value=5
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Individual", "Trustmark Dealer"]
    )

    owner = st.selectbox(
        "Owner Type",
        [
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    )

models = [
    "Maruti Swift",
    "Hyundai i20",
    "Hyundai Verna",
    "Honda City",
    "Toyota Innova",
    "Mahindra XUV500",
    "Ford EcoSport",
    "Renault Duster",
    "Other"
]

selected_model = st.selectbox(
    "Car Model",
    models
)

if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[0] * len(model_columns)],
        columns=model_columns
    )

    input_data["km_driven"] = km_driven
    input_data["car_age"] = car_age

    # Fuel
    fuel_col = f"fuel_{fuel}"
    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1

    # Seller
    seller_col = f"seller_type_{seller_type}"
    if seller_col in input_data.columns:
        input_data[seller_col] = 1

    # Transmission
    transmission_col = f"transmission_{transmission}"
    if transmission_col in input_data.columns:
        input_data[transmission_col] = 1

    # Owner
    owner_col = f"owner_{owner}"
    if owner_col in input_data.columns:
        input_data[owner_col] = 1

    # Model
    model_col = f"model_{selected_model}"
    if model_col in input_data.columns:
        input_data[model_col] = 1

    prediction_log = model.predict(input_data)

    prediction = np.expm1(prediction_log)

    st.success(
        f"Estimated Car Price: ₹{prediction[0]:,.0f}"
    )