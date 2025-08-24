import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Load the trained model, scaler, and column names
try:
    model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'xgb_model.joblib', 'scaler.joblib', and 'model_columns.joblib' are in the same directory.")
    st.stop()


# --- UI Configuration ---
st.set_page_config(page_title="Hospital Readmission Predictor", layout="wide")
st.title("🏥 Hospital Readmission Risk Predictor")
st.write("Enter patient details to predict the risk of readmission within 30 days.")

# --- Feature Input Columns ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Patient Demographics")
    race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])

with col2:
    st.header("Admission Details")
    time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 3)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
    num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

with col3:
    st.header("Medical History & Medications")
    number_inpatient = st.slider("Number of Inpatient Visits (Prev. Year)", 0, 21, 0)
    diag_1 = st.selectbox("Primary Diagnosis Category", ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'])
    insulin = st.selectbox("Insulin Prescription", ['No', 'Steady', 'Up', 'Down'])
    num_medications = st.slider("Number of Medications", 1, 81, 15)

# --- Prediction Logic ---
if st.button("Predict Readmission Risk", type="primary"):
    # 1. Create a DataFrame from the inputs in the exact format the model expects.
    # We use the saved `model_columns` for this.
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    # 2. Populate the user-provided feature values
    # Numerical features
    input_data['time_in_hospital'] = time_in_hospital
    input_data['num_lab_procedures'] = num_lab_procedures
    input_data['num_procedures'] = num_procedures
    input_data['num_medications'] = num_medications
    input_data['number_inpatient'] = number_inpatient
    input_data['number_diagnoses'] = number_diagnoses

    # --- CORRECTED CATEGORICAL FEATURE HANDLING ---
    # We check if the column for the user's choice exists in our model's columns.
    # If it does, we set it to 1. If not (because it was the category dropped
    # during training), we do nothing, and that's the correct behavior.

    # Race
    race_col = f'race_{race}'
    if race_col in input_data.columns:
        input_data[race_col] = 1

    # Gender
    gender_col = f'gender_{gender}'
    if gender_col in input_data.columns:
        input_data[gender_col] = 1

    # Age
    # Clean the age feature name to match the format used during training
    cleaned_age = re.sub(r"\[|\]|<", "_", age)
    age_col = f'age_{cleaned_age}'
    if age_col in input_data.columns:
        input_data[age_col] = 1

    # Diag_1
    diag_col = f'diag_1_{diag_1}'
    if diag_col in input_data.columns:
        input_data[diag_col] = 1

    # Insulin
    insulin_col = f'insulin_{insulin}'
    if insulin_col in input_data.columns:
        input_data[insulin_col] = 1
        
    # --- END OF CORRECTION ---

    # 3. Scale the numerical features using the saved scaler
    # Create a list of the numerical columns that need scaling
    cols_to_scale = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'num_medications'
    ]
    # Filter this list to only include columns that are actually in the input_data
    # (number_outpatient and number_emergency are not in our UI, so they will be 0 and scaled)
    cols_to_scale_existing = [col for col in cols_to_scale if col in input_data.columns]
    input_data[cols_to_scale_existing] = scaler.transform(input_data[cols_to_scale_existing])

    # 4. Make prediction
    prediction_proba = model.predict_proba(input_data)[:, 1]
    risk_score = prediction_proba[0]

    # 5. Display the result
    st.subheader("Prediction Result")
    # Using a threshold of 0.5 is standard, but for medical applications, a lower
    # threshold might be chosen to increase recall (catch more high-risk patients)
    if risk_score > 0.5:
        st.error(f"High Risk of Readmission (Risk Score: {risk_score:.2f})")
        st.warning("Follow-up care is strongly recommended for this patient.")
    else:
        st.success(f"Low Risk of Readmission (Risk Score: {risk_score:.2f})")
        st.info("Standard discharge protocol is likely sufficient.")