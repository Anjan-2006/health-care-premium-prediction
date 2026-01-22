import streamlit as st
from prediction_helper import predict


st.title("Health Care Premium Prediction")

# -------- Row 1 (Numerical) --------
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 18, 100, step=1)
with col2:
    num_dependants = st.number_input("Number of Dependants", 0, 20, step=1)
with col3:
    genetical_risk = st.number_input("Genetical Risk Score", 0, 5, step=1)

# -------- Row 2 --------
col4, col5, col6 = st.columns(3)
with col4:
    income_lakhs = st.number_input("Income (in Lakhs)", 0, 200, step=1)
with col5:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col6:
    marital_status = st.selectbox("Marital Status", ["Unmarried", "Married"])

# -------- Row 3 --------
col7, col8, col9 = st.columns(3)
with col7:
    region = st.selectbox(
        "Region",
        ["Northeast", "Northwest", "Southeast", "Southwest"]
    )
with col8:
    bmi_category = st.selectbox(
        "BMI Category",
        ["Underweight", "Normal", "Overweight", "Obesity"]
    )
with col9:
    smoking_status = st.selectbox(
        "Smoking Status",
        ["Regular", "Occasional", "No Smoking"]
    )

# -------- Row 4 --------
col10, col11, col12 = st.columns(3)
with col10:
    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-Employed", "Freelancer"]
    )
with col11:
   insurance_plan = st.selectbox(
        "Insurance Plan",
        ["Bronze", "Silver", "Gold"]
    )
with col12:
   medical_history = st.selectbox(
        "Medical History",
        [
            "No Disease",
            "Diabetes",
            "High blood pressure",
            "Heart disease",
            "Thyroid",
            "Diabetes & High blood pressure",
            "Diabetes & Heart disease",
            "High blood pressure & Heart disease",
            "Diabetes & Thyroid"
        ]
    )


# -------- Input Dictionary --------
input_dict = {
    "age": age,
    "number_of_dependants": num_dependants,
    "income_lakhs": income_lakhs,
    "genetical_risk": genetical_risk,
    "gender": gender,
    "marital_status": marital_status,
    "region": region,
    "bmi_category": bmi_category,
    "smoking_status": smoking_status,
    "employment_status": employment_status,
    "insurance_plan": insurance_plan,
    "medical_history": medical_history,
}

# -------- Predict Button --------
st.markdown("---")
if st.button("Predict"):
    prediction=predict(input_dict)
    st.success(f"The Predcited amount is {prediction}")