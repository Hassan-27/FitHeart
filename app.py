import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# Load the trained model and preprocessor (pipeline)
try:
    model_pipeline = joblib.load('heart_disease_model.pkl')
    success_message = st.empty()
    success_message.success("Model loaded successfully!")
    time.sleep(5)
    success_message.empty()
except FileNotFoundError:
    st.error("Error: heart_disease_model.pkl not found. Please run train.py first.")
    st.stop()

# Define the features based on the training script
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Streamlit App Title
st.set_page_config(page_title="FitHeart (Heart Disease Prediction)", layout="centered")
st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <div style="display: flex; align-items: center;">
        <i class="bi bi-heart-pulse-fill" style="font-size: 48px; color: #FF4B4B;"></i>
        <h1 style="margin-left: 16px; color: #FF4B4B;">FitHeart</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.write("Enter the patient's details to predict the risk of heart disease.")

# Create input widgets for each feature
with st.form("prediction_form"):
    st.header("Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 29, 77, 50)
        sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox("Chest Pain Type (cp)", [
            ("Typical Angina", 1),
            ("Atypical Angina", 2),
            ("Non-anginal Pain", 3),
            ("Asymptomatic", 4)
        ], format_func=lambda x: x[0])
        trestbps = st.slider("Resting Blood Pressure (trestbps)", 94, 200, 120)
        chol = st.slider("Cholesterol (chol)", 126, 564, 240)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [("False", 0), ("True", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [
            ("Normal", 0),
            ("ST-T wave abnormality", 1),
            ("Left ventricular hypertrophy", 2)
        ], format_func=lambda x: x[0])
        thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 71, 202, 150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        oldpeak = st.slider("ST depression induced by exercise relative to rest (oldpeak)", 0.0, 6.2, 1.0, 0.1)

    slope = st.selectbox("Slope of the peak exercise ST segment (slope)", [
        ("Upsloping", 1),
        ("Flat", 2),
        ("Downsloping", 3)
    ], format_func=lambda x: x[0])
    ca = st.selectbox("Number of major vessels (0-3) colored by flourosopy (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (thalassemia)", [
        ("Normal", 3),
        ("Fixed Defect", 6),
        ("Reversible Defect", 7)
    ], format_func=lambda x: x[0])

    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

    if submitted:
        try:
            # Create a DataFrame from the input values
            feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_data = pd.DataFrame([[age, sex[1], cp[1], trestbps, chol, fbs[1], restecg[1], thalach, exang[1], oldpeak, slope[1], ca, thal[1]]],
                                      columns=feature_columns)

            # Handle data types properly
            for col in numerical_features:
                input_data[col] = input_data[col].astype('float64')
            
            for col in categorical_features:
                input_data[col] = input_data[col].astype(str)
            
            # Ensure no missing values
            if input_data.isnull().any().any():
                st.error("Error: Missing values detected in input data.")
                st.stop()
            
            # Make prediction
            prediction = model_pipeline.predict(input_data)[0]
            prediction_proba = model_pipeline.predict_proba(input_data)[0]
            
            st.markdown("--- ")
            st.subheader("🎯 Prediction Results:")
            
            # Ensure we have the right probability interpretation
            classes = model_pipeline.classes_
            
            # Find probabilities for each class
            if len(classes) == 2:
                if 0 in classes and 1 in classes:
                    # Standard binary: 0 = No Disease, 1 = Disease
                    prob_no_disease = prediction_proba[list(classes).index(0)]
                    prob_disease = prediction_proba[list(classes).index(1)]
                else:
                    # If classes are different, use first as no disease, second as disease
                    prob_no_disease = prediction_proba[0]
                    prob_disease = prediction_proba[1]
            else:
                st.error(f"Unexpected number of classes: {len(classes)}")
                st.stop()
            
            if prob_disease > 0.5:
                st.error(f"🚨 **Prediction: Heart Disease Present**")
                st.write(f"**Confidence:** {prob_disease * 100:.2f}%")
                st.warning("💡 **Recommendation:** Please consult a cardiologist for further evaluation.")

                if prob_disease > 0.75:
                    risk_level = "Very High Risk"
                    risk_color = "🔴"
                    risk_description = "Very high probability of heart disease. Immediate medical consultation strongly recommended."
                elif prob_disease > 0.6:
                    risk_level = "High Risk"
                    risk_color = "🔴"
                    risk_description = "High probability of heart disease. Medical consultation recommended."
                else:
                    risk_level = "Moderate Risk"
                    risk_color = "🟡"
                    risk_description = "Moderate probability of heart disease. Consider lifestyle modifications and medical consultation."
            else:
                st.success(f"✅ **Prediction: No Heart Disease**")
                st.write(f"**Confidence:** {prob_no_disease * 100:.2f}%")
                st.info("💡 **Recommendation:** Maintain a healthy lifestyle and get regular check-ups.")

                if prob_no_disease > 0.75:
                    risk_level = "Very Low Risk"
                    risk_color = "🟢"
                    risk_description = "Very low probability of heart disease. Maintain current healthy lifestyle."
                elif prob_no_disease > 0.6:
                    risk_level = "Low Risk"
                    risk_color = "🟢"
                    risk_description = "Low probability of heart disease. Continue healthy habits and regular check-ups."
                else:
                    risk_level = "Slight Risk"
                    risk_color = "🟡"
                    risk_description = "Slight probability of heart disease. Continue to monitor your health and consult a doctor if you have any concerns."

            
            st.markdown("---")
            st.subheader("⚕️ Risk Assessment:")
            st.write(f"**{risk_color} {risk_level}**")
            st.write(f"*{risk_description}*")
            
            # Additional insights based on input values
            st.subheader("📋 Clinical Insights:")
            
            risk_factors = []
            protective_factors = []
            
            # Age factor
            if age >= 65:
                risk_factors.append("Advanced age (≥65 years)")
            elif age <= 40:
                protective_factors.append("Younger age (≤40 years)")
            
            # Gender factor
            if sex[1] == 1 and age >= 45:  # Male over 45
                risk_factors.append("Male gender with age ≥45")
            elif sex[1] == 0 and age >= 55:  # Female over 55
                risk_factors.append("Female gender with age ≥55")
            
            # Chest pain
            if cp[1] == 1:  # Typical angina
                risk_factors.append("Typical angina chest pain")
            elif cp[1] == 4:  # Asymptomatic
                protective_factors.append("No chest pain symptoms")
            
            # Blood pressure
            if trestbps >= 140:
                risk_factors.append("High blood pressure (≥140 mmHg)")
            elif trestbps <= 120:
                protective_factors.append("Normal blood pressure (≤120 mmHg)")
            
            # Cholesterol
            if chol >= 240:
                risk_factors.append("High cholesterol (≥240 mg/dl)")
            elif chol <= 200:
                protective_factors.append("Normal cholesterol (≤200 mg/dl)")
            
            # Exercise capacity
            if thalach <= 100:
                risk_factors.append("Poor exercise capacity (max HR ≤100)")
            elif thalach >= 160:
                protective_factors.append("Good exercise capacity (max HR ≥160)")
            
            # Exercise induced angina
            if exang[1] == 1:
                risk_factors.append("Exercise-induced chest pain")
            else:
                protective_factors.append("No exercise-induced chest pain")
            
            # Display factors
            if risk_factors:
                st.write("**🔴 Risk Factors Present:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            
            if protective_factors:
                st.write("**🟢 Protective Factors:**")
                for factor in protective_factors:
                    st.write(f"• {factor}")
            
            # Debug information (can be removed in production)
            # with st.expander("🔧 Debug Information"):
            #     st.write(f"Raw prediction: {prediction}")
            #     st.write(f"Raw disease probability: {prob_disease:.4f}")
            #     st.write(f"Raw no disease probability: {prob_no_disease:.4f}")
            #     st.write(f"Corrected disease risk: {actual_disease_risk:.4f}")
            #     st.write(f"Model classes: {model_pipeline.classes_}")
            #     st.write("**Note:** Model appears to have inverted predictions. Using corrected logic.")
            #     st.write("**Input values:**")
            #     for col, val in zip(feature_columns, input_data.iloc[0]):
            #         st.write(f"{col}: {val}")
                    
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("**Error details:**")
            st.exception(e)

# Sidebar with information
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This application uses machine learning to predict heart disease risk based on clinical parameters from the UCI Cleveland Heart Disease dataset.

**Key Features:**
- Binary classification (Disease/No Disease)
- Class-balanced model training
- Probability-based predictions
- Risk level assessment
- Clinical insights

**Input Parameters:**
- **Age**: Patient age in years
- **Sex**: Male/Female
- **CP**: Chest pain type (1-4)
- **Trestbps**: Resting blood pressure
- **Chol**: Serum cholesterol
- **FBS**: Fasting blood sugar > 120 mg/dl
- **Restecg**: Resting ECG results
- **Thalach**: Maximum heart rate achieved
- **Exang**: Exercise induced angina
- **Oldpeak**: ST depression
- **Slope**: ST segment slope
- **CA**: Major vessels colored by fluoroscopy
- **Thal**: Thalassemia test results

**Note:** This version includes corrected prediction logic 
to handle model calibration issues.
""")

st.sidebar.header("🚨 Important Note")
st.sidebar.warning("""
This tool is for educational and informational purposes only. 

**DO NOT** use this as a substitute for professional medical advice, diagnosis, or treatment. 

Always consult qualified healthcare providers for medical decisions.

⚠️ **Limitation:** Predictions may not be highly reliable due to the relatively small dataset used for training the model. 
""")

st.markdown("---")
st.caption("""
💡 **Disclaimer:** This application is for educational and informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.  
⚠️ Predictions may also be affected by the small dataset size, so results should be interpreted with caution.
""")