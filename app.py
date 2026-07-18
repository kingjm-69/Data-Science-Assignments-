import streamlit as st
import pickle
import pandas as pd

# ---------------------------------------
# Load Trained Model and Scaler
# ---------------------------------------
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------
# Title
# ---------------------------------------
st.title("🩺 Diabetes Prediction System")

st.markdown("""
This application predicts whether a patient is likely to have diabetes
using a **Logistic Regression Machine Learning Model**.

Please enter the patient's medical details below.
""")

st.divider()

# ---------------------------------------
# Input Fields
# ---------------------------------------
st.subheader("Enter Patient Details")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    format="%.2f"
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.50,
    format="%.2f"
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

st.divider()

# ---------------------------------------
# Prediction Button
# ---------------------------------------
if st.button("Predict Diabetes"):

    # Create DataFrame
    user_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Scale Data
    scaled_data = scaler.transform(user_data)

    # Prediction
    prediction = model.predict(scaled_data)

    # Prediction Probability
    probability = model.predict_proba(scaled_data)

    diabetic_probability = probability[0][1] * 100
    non_diabetic_probability = probability[0][0] * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Patient is likely to have Diabetes")
    else:
        st.success("✅ Patient is NOT likely to have Diabetes")

    st.write("### Prediction Probability")

    st.progress(diabetic_probability / 100)

    st.write(f"**Diabetic Probability:** {diabetic_probability:.2f}%")
    st.write(f"**Non-Diabetic Probability:** {non_diabetic_probability:.2f}%")

    st.divider()

    with st.expander("View Entered Details"):

        st.dataframe(user_data)