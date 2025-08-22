import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Dry Bean Species Classifier",
    page_icon="🫘",
    layout="wide"
)

# --- Load Model, Scaler, and Encoders ---


@st.cache_resource
def load_artifacts():
    with open('bean_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('scaler.pkl', 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
    with open('label_encoder.pkl', 'rb') as f_encoder:
        label_encoder = pickle.load(f_encoder)
    with open('features.pkl', 'rb') as f_features:
        features = pickle.load(f_features)
    with open('feature_medians.pkl', 'rb') as f_medians:
        medians = pickle.load(f_medians)
    return model, scaler, label_encoder, features, medians


model, scaler, label_encoder, features, medians = load_artifacts()

# --- App Title and Description ---
st.title("🫘 Dry Bean Species Classifier")
st.markdown(
    "Enter the physical measurements of a dry bean to predict its species.")
st.markdown("---")

# --- User Input ---
st.sidebar.header("Bean Measurements")

# Dictionary to hold user inputs, initialized with median values
input_data = medians.to_dict()

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Primary Measurements")
    input_data['Area'] = st.number_input("Area", value=float(medians['Area']))
    input_data['Perimeter'] = st.number_input(
        "Perimeter", value=float(medians['Perimeter']))
    input_data['MajorAxisLength'] = st.number_input(
        "Major Axis Length", value=float(medians['MajorAxisLength']))
    input_data['MinorAxisLength'] = st.number_input(
        "Minor Axis Length", value=float(medians['MinorAxisLength']))
    input_data['ConvexArea'] = st.number_input(
        "Convex Area", value=float(medians['ConvexArea']))
    input_data['EquivDiameter'] = st.number_input(
        "Equivalent Diameter", value=float(medians['EquivDiameter']))

with col2:
    st.subheader("Shape Ratios")
    input_data['AspectRation'] = st.number_input(
        "Aspect Ratio", value=float(medians['AspectRation']))
    input_data['Eccentricity'] = st.number_input(
        "Eccentricity", value=float(medians['Eccentricity']))
    input_data['Extent'] = st.number_input(
        "Extent", value=float(medians['Extent']))
    input_data['Solidity'] = st.number_input(
        "Solidity", value=float(medians['Solidity']))
    input_data['roundness'] = st.number_input(
        "Roundness", value=float(medians['roundness']))
    input_data['Compactness'] = st.number_input(
        "Compactness", value=float(medians['Compactness']))

with col3:
    st.subheader("Derived Shape Factors")
    input_data['ShapeFactor1'] = st.number_input(
        "Shape Factor 1", value=float(medians['ShapeFactor1']))
    input_data['ShapeFactor2'] = st.number_input(
        "Shape Factor 2", value=float(medians['ShapeFactor2']))
    input_data['ShapeFactor3'] = st.number_input(
        "Shape Factor 3", value=float(medians['ShapeFactor3']))
    input_data['ShapeFactor4'] = st.number_input(
        "Shape Factor 4", value=float(medians['ShapeFactor4']))

# --- Prediction Logic ---
if st.button("Classify Bean", type="primary"):

    # 1. Convert user input to a DataFrame in the correct order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]

    # 2. Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # 3. Make a prediction (returns a number, e.g., 0, 1, 2...)
    prediction_encoded = model.predict(input_scaled)[0]

    # 4. Convert the encoded prediction back to the species name
    prediction_species = label_encoder.inverse_transform([prediction_encoded])[
        0]

    # --- Display the result ---
    st.markdown("---")
    st.subheader("Prediction Result")
    st.success(f"### The predicted bean species is: **{prediction_species}**")

st.markdown("---")
st.info("""
**About this model:**  
This dry bean species classifier uses a machine learning model trained on a labeled dataset of dry bean measurements. The model analyzes various physical characteristics to predict the bean species.

**Note:** This tool is for informational and educational purposes only.should not be used for critical decisions.
""")
