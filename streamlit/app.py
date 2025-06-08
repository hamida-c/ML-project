import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Function to load pickle files
@st.cache_data
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load all required files
model = load_pickle("model.pkl")
scaler = load_pickle("scaler.pkl")
encoder_gender = load_pickle("gender.pkl")
encoder_job_level = load_pickle("job_level.pkl")
encoder_job_satisfaction = load_pickle("job_satisfaction.pkl")
encoder_marital_status = load_pickle("marital_status.pkl")
encoder_overtime = load_pickle("overtime.pkl")
encoder_remote = load_pickle("remote_work.pkl")
encoder_education = load_pickle("education_level.pkl")
encoder_wlb = load_pickle("work_life_balance.pkl")

# Display banner image
image = Image.open("employee-churn.png")  # Your uploaded image
st.image(image, caption="Employee Attrition Predictor", use_container_width=True)

# App title
st.title("🔍 Employee Attrition Prediction")
st.write("Fill in the details below to predict the likelihood of attrition.")

# Define readable categories
job_level_options = ["Entry", "Mid", "Senior"]
job_satisfaction_options = ["Low", "Medium", "High", "Very High"]  # Ordinal
work_life_balance_options = ["Poor", "Fair", "Good", "Excellent"]  # Ordinal
allowed_education_labels = ['Associate Degree', 'Masters Degree', 'Bachelors Degree', 'High School', 'PhD']

# User input section with readable labels
gender = st.selectbox("Gender", ["Male", "Female"])
job_level = st.selectbox("Job Level", job_level_options)
job_satisfaction = st.selectbox("Job Satisfaction", job_satisfaction_options)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
overtime = st.selectbox("Overtime", ["Yes", "No"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
education = st.selectbox("Education Level", allowed_education_labels)
work_life_balance = st.selectbox("Work-Life Balance", work_life_balance_options)

# Numeric fields
age = st.number_input("Age", min_value=18, max_value=95, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=8000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)

# NEW numeric inputs
number_of_promotions = st.number_input("Number of Promotions", min_value=0, max_value=20, value=0)
number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

# Encode categorical fields with the encoders
gender_encoded = encoder_gender.transform([gender])[0]
job_level_encoded = encoder_job_level.transform([job_level])[0]
job_sat_encoded = encoder_job_satisfaction.transform([job_satisfaction])[0]
marital_encoded = encoder_marital_status.transform([marital_status])[0]
overtime_encoded = encoder_overtime.transform([overtime])[0]
remote_encoded = encoder_remote.transform([remote_work])[0]

# Validate education level before encoding
if education not in encoder_education.classes_:
    st.error(f"Invalid education level: '{education}'. Please select from the allowed options.")
    st.stop()
else:
    education_encoded = encoder_education.transform([education])[0]

wlb_encoded = encoder_wlb.transform([work_life_balance])[0]

# Combine inputs in the order your model expects:
input_data = np.array([[age, gender_encoded, job_level_encoded, job_sat_encoded,
                        marital_encoded, overtime_encoded, remote_encoded,
                        education_encoded, wlb_encoded, monthly_income, years_at_company,
                        number_of_promotions, number_of_dependents]])

# Standardize
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Attrition Risk"):
    result = model.predict(scaled_input)[0]
    if result == 1:
        st.error("🚨 This employee is at risk of leaving.")
    else:
        st.success("✅ This employee is likely to stay.")
