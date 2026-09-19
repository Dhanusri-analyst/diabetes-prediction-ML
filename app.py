import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 Diabetes Prediction System")
st.write(
    "Enter the patient information below to generate a machine-learning prediction."
)

st.divider()

# Input fields
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    max_value=250.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)

# Prediction button
if st.button("Predict Diabetes", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Convert zero values to missing values,
    # matching the preprocessing used during training
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[zero_as_missing] = input_data[zero_as_missing].replace(
        0, np.nan
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Model Prediction: Diabetes")
    else:
        st.success("✅ Model Prediction: No Diabetes")

    st.write(
        f"Estimated model probability of diabetes: **{probability:.2%}**"
    )

    st.caption(
        "This application is for educational purposes and is not a medical diagnosis."
    )