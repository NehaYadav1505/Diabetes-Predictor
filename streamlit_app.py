import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

st.title("🩺 Diabetes Predictor")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    result = "🩸 Diabetes Detected!" if prediction[0] == 1 else "💚 No Diabetes"
    st.success(result)
