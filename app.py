import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page Configuration
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

# 1. Load the trained model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model files not found. Please run 'train_model.py' first.")
    st.stop()

# 2. Header
st.title("🔬 Breast Cancer Prediction System")
st.markdown("This AI-powered tool predicts the likelihood of a tumor being **Benign** or **Malignant**.")

# 3. Sidebar Inputs
st.sidebar.header("Step 1: Input Cell Features")
st.sidebar.info("Adjust the sliders below based on cytological analysis.")

def user_input_features():
    # We use the same 10 features used in training
    radius = st.sidebar.slider('Mean Radius', 6.0, 30.0, 14.0)
    texture = st.sidebar.slider('Mean Texture', 9.0, 40.0, 19.0)
    perimeter = st.sidebar.slider('Mean Perimeter', 43.0, 190.0, 90.0)
    area = st.sidebar.slider('Mean Area', 143.0, 2500.0, 650.0)
    smoothness = st.sidebar.slider('Mean Smoothness', 0.05, 0.25, 0.1)
    compactness = st.sidebar.slider('Mean Compactness', 0.01, 0.35, 0.1)
    concavity = st.sidebar.slider('Mean Concavity', 0.0, 0.45, 0.1)
    concave_points = st.sidebar.slider('Mean Concave Points', 0.0, 0.20, 0.05)
    symmetry = st.sidebar.slider('Mean Symmetry', 0.1, 0.3, 0.2)
    fractal_dim = st.sidebar.slider('Mean Fractal Dim', 0.05, 0.1, 0.06)

    data = {
        'mean radius': radius, 'mean texture': texture, 'mean perimeter': perimeter,
        'mean area': area, 'mean smoothness': smoothness, 'mean compactness': compactness,
        'mean concavity': concavity, 'mean concave points': concave_points,
        'mean symmetry': symmetry, 'mean fractal dimension': fractal_dim
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Main Panel
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Data Overview")
    st.dataframe(input_df)

    if st.button("Run Diagnosis", type="primary"):
        # Scale the input using the loaded scaler
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # The dataset target: 0 = Malignant, 1 = Benign
        # We invert this logic for display to make it clearer
        is_malignant = prediction[0] == 0
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        if is_malignant:
            st.error(f"⚠️ **Result: MALIGNANT (Cancerous)**")
            confidence = prediction_proba[0][0] # Probability of 0
        else:
            st.success(f"✅ **Result: BENIGN (Safe)**")
            confidence = prediction_proba[0][1] # Probability of 1
            
        st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")
        
        st.info("Note: This is an AI-assisted prediction and should not replace professional medical diagnosis.")

with col2:
    st.subheader("Model Insights")
    st.write("The model analyzes specific features of the cell nuclei.")
    st.markdown("""
    - **Radius**: Size of the cell
    - **Texture**: Standard deviation of gray-scale values
    - **Smoothness**: Variation in radius lengths
    """)