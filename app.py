import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import os
import pickle
# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct the path to the pickle file relative to the current script
pickle_file_path = os.path.join(current_dir, "diabetic_prediction.pkl")
with open(pickle_file_path, "rb") as f:
    model = pickle.load(f)
# Create a Streamlit application
st.title("Diabetes Prediction App")

# Input fields for user data
pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
glucose = st.number_input("Glucose", min_value=0, step=1, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1, value=0)
insulin = st.number_input("Insulin", min_value=0, step=1, value=0)
bmi = st.number_input("BMI", min_value=0.0, step=0.1, value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.001, value=0.0)
age = st.number_input("Age", min_value=0, step=1, value=0)

# Prediction button
if st.button("Predict"):
    # Prepare the input data as a pandas DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age]
    })
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    if prediction[0] == 1:
        st.success("The model predicts that the person has diabetes.")
    else:
        st.success("The model predicts that the person does not have diabetes.")

# Run the application by executing the script
# Save this script as app.py, and then run it using the command:
# streamlit run app.py
