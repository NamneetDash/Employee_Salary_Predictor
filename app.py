# app.py

import streamlit as st

import pandas as pd
import joblib
import numpy as np
import base64 # Added for image encoding

# --- Helper Functions ---

@st.cache_data
def get_image_as_base64(file_path):
    """Reads an image file and returns its Base64 encoded version."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_page_background(image_file):
    """Sets the background of the Streamlit page with a dark overlay."""
    base64_img = get_image_as_base64(image_file)
    if base64_img:
        # This CSS adds a 60% opaque black gradient on top of the image
        page_bg_img_style = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("data:image/jpg;base64,{base64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img_style, unsafe_allow_html=True)

def load_external_css(file_path):
    """Loads an external CSS file and injects it into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {file_path}. Please ensure the file is in the correct directory.")
        st.stop()

def preprocess_input(df_raw, encoders, expected_columns):
    """
    Preprocesses the raw input DataFrame to be ready for the model.
    This includes label encoding and ensuring column order.
    """
    df = df_raw.copy()
    
    # Apply label encoding
    for col_name, encoder in encoders.items():
        if col_name in df.columns:
            known_classes = list(encoder.classes_)
            df[col_name] = df[col_name].apply(lambda x: encoder.transform([x])[0] if x in known_classes else 0)
        else:
            st.warning(f"Column '{col_name}' expected for encoding but not found in input data. This may affect prediction accuracy.")
    
    # Ensure all columns are numeric, coercing errors
    for col_name in df.columns:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0) # Added fillna(0) for robustness

    # Reorder columns to match the model's training order
    return df[expected_columns]

# --- Load Model and Encoders ---
try:
    model_pipeline = joblib.load("best_model_pipeline.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    income_encoder = joblib.load("income_encoder.pkl") # To decode predictions
except FileNotFoundError:
    st.error("Error: Model or encoder files not found. Please ensure 'best_model_pipeline.pkl', 'label_encoders.pkl', and 'income_encoder.pkl' are in the same directory as the app.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Professional Employee Salary Predictor 💼",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Set Background and Load Custom CSS ---
set_page_background("background.jpg") 
load_external_css("custom_styles.css")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "About the Project"])


# --- Page Routing ---
if page == "Predictor":
    # --- Main Title and Description ---
    st.title("💸 Employee Salary Predictor")
    st.markdown("""
    <p class='intro-text'>
        Uncover powerful insights into potential income brackets. This advanced analytical tool
        leverages a robust machine learning model to predict whether an individual's
        annual income is <span style='font-weight:bold; color: #82aaff;'>exceeding $50,000</span> or
        <span style='font-weight:bold; color: #ff8080;'>$50,000 or less</span>, based on a comprehensive
        set of demographic and employment attributes.
    </p>
    """, unsafe_allow_html=True)

    # --- Input Section ---
    st.header("⚙️ Employee Details for Prediction")
    st.markdown("Please provide the following information to get a salary class prediction.")

    EXPECTED_COLUMNS_ORDER = [
        'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country'
    ]

    with st.container(border=True):
        st.markdown("### 🧑‍💼 Personal & Work Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 17, 75, 30, help="Age of the individual (17-75).")
            race = st.selectbox("Race", label_encoders['race'].classes_, help="Racial background of the individual.")
        with col2:
            workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_, help="Type of employer or work sector.")
            marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_, help="Marital status of the individual.")
        with col3:
            gender = st.selectbox("Gender", label_encoders['gender'].classes_, help="Gender of the individual.")
            relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_, help="Family relationship status.")

    with st.container(border=True):
        st.markdown("### 🎓 Educational & Professional Details")
        col1, col2, col3 = st.columns(3)
        education_to_num = { "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16 }
        with col1:
            education_display_options = sorted(education_to_num.keys(), key=lambda x: education_to_num[x])
            selected_education_str = st.selectbox("Education Level", education_display_options, index=education_display_options.index("Bachelors"), help="Highest level of education achieved.")
            educational_num = education_to_num[selected_education_str]
        with col2:
            occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_, help="The type of occupation.")
        with col3:
            hours_per_week = st.slider("Hours per Week", 1, 80, 40, help="Number of hours worked per week (1-80).")

    with st.container(border=True):
        st.markdown("### 💰 Financial & Geographic Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, help="Amount of capital gains (e.g., from investments).")
        with col2:
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0, help="Amount of capital losses (e.g., from investments).")
        with col3:
            native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_, index=list(label_encoders['native-country'].classes_).index('United-States'), help="Country of origin.")
        fnlwgt = st.number_input("Fnlwgt (Final Weight)", min_value=10000, max_value=1500000, value=200000, help="The number of people the census believes the entry represents. This is a statistical weight.")

    # --- Prediction Section ---
    st.markdown("---")
    _, predict_col, _ = st.columns([1, 2, 1])
    if predict_col.button("🚀 Predict Salary Class", use_container_width=True):
        input_data = pd.DataFrame([{'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'educational-num': educational_num, 'marital-status': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss, 'hours-per-week': hours_per_week, 'native-country': native_country}])
        processed_input = preprocess_input(input_data, label_encoders, EXPECTED_COLUMNS_ORDER)
        prediction_numeric = model_pipeline.predict(processed_input)[0]
        prediction_class = income_encoder.inverse_transform([prediction_numeric])[0]
        color = '#28a745' if prediction_class == '>50K' else '#ff8080'
        st.markdown(f"<div class='prediction-result'>Predicted Income Class: <span style='color: {color};'>{prediction_class}</span></div>", unsafe_allow_html=True)

    # --- Batch Prediction Section ---
    st.markdown("---")
    st.header("📁 Batch Prediction (Upload CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        batch_data_raw = pd.read_csv(uploaded_file)
        missing_cols = [col for col in EXPECTED_COLUMNS_ORDER if col not in batch_data_raw.columns]
        if missing_cols:
            st.error(f"Error: CSV is missing required columns: {', '.join(missing_cols)}.")
        else:
            batch_processed = preprocess_input(batch_data_raw, label_encoders, EXPECTED_COLUMNS_ORDER)
            predictions = income_encoder.inverse_transform(model_pipeline.predict(batch_processed))
            results_df = batch_data_raw.copy()
            results_df['Predicted_Income_Class'] = predictions
            st.dataframe(results_df.head(), use_container_width=True)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions as CSV", csv, "predicted_salaries.csv", "text/csv", use_container_width=True)

elif page == "About the Project":
    st.title("💡 About the Project")
    st.markdown("""
    This project is a capstone submission for my **IBM Skillsbuild** journey, focusing on
    practical applications of Machine Learning and Data Science. It meticulously demonstrates
    a complete end-to-end analytical pipeline: from initial data cleaning and preprocessing,
    to advanced predictive model training, and finally, robust deployment into an interactive,
    user-friendly web application. The primary objective is to provide a reliable tool for
    predicting income levels based on publicly available demographic data.
    """)

    st.header("📖 How to Use the Income Predictor")
    st.markdown("""
    1.  **Navigate to the Predictor:** Use the sidebar menu to select the "Predictor" page.
    2.  **Input Employee Details:** Utilize the intuitive sliders, precise dropdown menus, and convenient number inputs to accurately provide the characteristics of the individual you wish to analyze.
    3.  **Get Prediction:** Once all details are entered, click the prominent "🚀 Predict Salary Class" button to see the result.
    4.  **Batch Prediction:** For bulk analyses, upload a CSV file using the "Batch Prediction" feature. Ensure your file has the **exact 13 required column names** for seamless processing.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    🚀 Built with ❤️ as part of the <strong>IBM SkillsBuild</strong> Project.
</div>
""", unsafe_allow_html=True)