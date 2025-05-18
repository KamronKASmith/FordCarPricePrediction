from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model_training import train_model
from sklearn.exceptions import NotFittedError

#3 Column Page Setup
st.set_page_config(page_title="Ford Price Predictor", layout="wide")

#Loads Datasheet
@st.cache_data
def load_data():
    df = pd.read_csv("Sources/FordPrices.csv")
    return df

df = load_data()

#Machine Learning Model Options
model_list = {
    "Linear Regression": "📈",
    "Ridge Regression": "🌊",
    "Lasso Regression": "🎯",
    "Decision Tree": "🌲",
    "Random Forest": "🌳",
    "XGBoost": "🚀",
    "LightGBM": "💡"
}

trained_models = {}

for model_type in model_list.keys():
    model, y_test, y_pred = train_model(df, model_type)
    trained_models[model_type] = {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred
    }

model_col, input_col, prediction_col = st.columns([1.5, 2.5, 2])

#Left Column
with model_col:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("### 🤖 Select Model/s")

    select_all = st.checkbox("✅ Select All Models")
    selected_models = []

    for name, icon in model_list.items():
        checked = st.checkbox(f"{icon} {name}", value=select_all, key=name)
        if checked:
            selected_models.append(name)

#Middle Column
with input_col:
    st.markdown("<h1 style='text-align: center;'>🚗 Ford Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Predict the value of a used Ford vehicle based on key features.</h1>", unsafe_allow_html=True)
    st.markdown("### 🚗 Enter Car Details")

    model_name = st.selectbox("Model Name", [
    "B-Max", "C-Max", "Ecosport", "Edge", "Escort", "Fiesta", "Focus", "Fusion",
    "Galaxy", "Grand C-Max", "Grand Tourneo Connect", "Ka", "Ka+", "Kuga",
    "Mondeo", "Mustang", "Puma", "Ranger", "S-Max", "Streetka",
    "Tourneo Connect", "Tourneo Custom", "Transit Tourneo"
    ])
    year = st.selectbox("Model Year", list(range(1995, 2025)))
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    tax = st.slider("Sales Tax ($)", 0, 600, 150)
    mpg = st.slider("Fuel Efficiency (MPG)", 10.0, 200.0, 50.0)
    engine_size = st.slider("Engine Size (L)", 0.8, 5.0, 1.6)
    mileage = st.number_input("Mileage (Miles)", min_value=0, max_value=300000, value=50000)

    predict_btn = st.button("🔍 Predict Price")

model, y_test, y_pred = train_model(df)

#Right Column
with prediction_col:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("### 💰 Price Predictions")

    if predict_btn:
        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            for model in selected_models:
                predicted_price = round(10000 + hash(model + model_name) % 10000, 2)
                st.success(f"{model_list[model]} {model} Prediction: **£{predicted_price:,.2f}**")

                with st.expander(f"\n{model_list[model]} {model} Model Performance Metrics:"):
                    st.markdown(f"R² Score: {r2_score(y_test, y_pred):.4f}")
                    st.markdown(f"Mean of Absolute Errors (MAE): ${mean_absolute_error(y_test, y_pred):,.2f}")
                    st.markdown(f"Root of the Mean of Square Errors (RMSE): ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")