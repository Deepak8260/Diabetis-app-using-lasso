import streamlit as st
import pickle
import numpy as np
import sklearn

# Load the trained model and scaler
with open("lasso_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define feature names
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

st.title("Diabetes Progression Prediction")
st.write("Enter the values for each feature to predict the target.")

# Input fields for features
inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# Convert inputs to numpy array and reshape for scaling
input_array = np.array(inputs).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Predict button
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Target Value: {prediction:.2f}")
