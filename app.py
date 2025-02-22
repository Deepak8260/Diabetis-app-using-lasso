import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("lasso_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define feature names
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

st.title("Diabetes Progression Prediction")
st.write("Enter the values for each feature to predict the target.")

# Arrange input fields into two columns
col1, col2 = st.columns(2)

inputs = []
for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        value = col1.number_input(f"{feature}", value=0.0)
    else:
        value = col2.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# Convert inputs to numpy array and reshape for scaling
input_array = np.array(inputs).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Predict button
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Target Value: {prediction:.2f}")
st.write("Built by Deepak")