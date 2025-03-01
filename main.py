import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64


# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoders
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

# Function to encode the image as Base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Function to set background with a dark overlay
def set_bg(image_file):
    base64_image = get_base64(image_file)
    bg_image_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0, 0.6)),  
                    url("data:image/jpg;base64,{base64_image}");
        background-size: cover;  
        background-position: center;  
        background-repeat: no-repeat;  
        background-attachment: fixed;  
        font-family: 'Poppins', sans-serif;
    }}

    h1 {{
        color: #ffffff;  
        text-align: center;
        font-weight: 600;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
         font-size: 34px;
    }}

    h2 {{
        text-align: center;
        font-size: 34px;
        font-weight: 600;
    }}

    p, label, .stSelectbox, .stNumber_input, .stButton {{
        font-size: 20px !important;
        font-weight: 400;
        color: white !important;
    }}

    .stButton>button {{
        background-color: #ff5722 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        transition: all 0.3s ease-in-out;
    }}

    .stButton>button:hover {{
        background-color: #e64a19 !important;
        transform: scale(1.05);
    }}
    </style>
    """
    st.markdown(bg_image_css, unsafe_allow_html=True)

# Call function with your image path
set_bg("download.jpg")

# --- Styled Title ---
st.markdown(
    "<h1>🧠 Autism Screening Prediction 🧠</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center; font-size: 20px;'>Enter patient details below to predict the likelihood of Autism Spectrum Disorder (ASD).</p>",
    unsafe_allow_html=True,
)

# --- Input Section ---
st.markdown("### 📌 Patient Details")
age = st.number_input("👶 Age", min_value=1, max_value=100, step=1)
gender = st.selectbox("⚧ Gender", ["Male", "Female"])
ethnicity = st.selectbox("🌍 Ethnicity", encoders.get("ethnicity", pd.Series()).classes_ if "ethnicity" in encoders else ["Unknown"])
jaundice = st.selectbox("🟡 Jaundice (At Birth)", ["Yes", "No"])
autism = st.selectbox("👨‍👩‍👦 Family Autism History", ["Yes", "No"])
country = st.selectbox("🏠 Country of Residence", encoders.get("contry_of_res", pd.Series()).classes_ if "contry_of_res" in encoders else ["Unknown"])

# --- Autism Questionnaire ---
st.markdown("### 📝 Autism Screening Questions")
score_descriptions = {
    1: "Prefers being alone over socializing",
    2: "Avoids or struggles with eye contact",
    3: "Strong preference for routines",
    4: "Struggles with understanding sarcasm/metaphors",
    5: "Overreacts to loud noises, bright lights, or textures",
    6: "Engages in repetitive movements or speech",
    7: "Difficulty understanding non-verbal cues",
    8: "Difficulty interpreting social norms",
    9: "Intense focus on specific topics",
    10: "Delayed speech development or limited verbal communication",
    11: "Difficulty making and maintaining friendships",
}

scores = [
    1 if st.selectbox(f"🔹 {desc}", ["Yes", "No"], index=0) == "Yes" else 0
    for i, desc in score_descriptions.items()
]

used_app_before = st.selectbox("📱 Used Screening App Before?", ["Yes", "No"])
relation = st.selectbox("👨‍👩‍👦 Relation to Patient", encoders.get("relation", pd.Series()).classes_ if "relation" in encoders else ["Unknown"])

# --- Label Encoding ---
label_mappings = {
    "gender": {"Male": "m", "Female": "f"},
    "jaundice": {"Yes": "yes", "No": "no"},
    "austim": {"Yes": "yes", "No": "no"},
    "used_app_before": {"Yes": "yes", "No": "no"},
}

def encode_input(column, value):
    if column in label_mappings and value in label_mappings[column]:
        value = label_mappings[column][value]
    if column not in encoders:
        return None
    if value in encoders[column].classes_:
        return encoders[column].transform([value])[0]
    else:
        return -1

gender = encode_input("gender", gender)
ethnicity = encode_input("ethnicity", ethnicity)
jaundice = encode_input("jaundice", jaundice)
autism = encode_input("austim", autism)
country = encode_input("contry_of_res", country)
used_app_before = encode_input("used_app_before", used_app_before)
relation = encode_input("relation", relation)

# --- Model Prediction ---
input_data = np.array([age] + scores + [gender, ethnicity, jaundice, autism, country, used_app_before, relation])

expected_features = model.n_features_in_
actual_features = input_data.shape[0]

if actual_features != expected_features:
    st.error(f"⚠️ Feature mismatch: Model expects {expected_features} features but received {actual_features}.")
    st.stop()

input_data = input_data.reshape(1, -1)

# --- Prediction Section ---
st.markdown("### 🔍 Prediction Result")
if st.button("🔮 Predict", use_container_width=True):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            threshold = 0.2
            prediction = 1 if proba[1] > threshold else 0
            confidence = proba[1]
        else:
            prediction = model.predict(input_data)[0]
            confidence = None

        if prediction == 1:
            st.markdown(
                "<h2 style='color: red;'>🛑 Positive for ASD</h2>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<h2 style='color: green;'>✅ Negative for ASD</h2>",
                unsafe_allow_html=True,
            )

        if confidence is not None:
            st.markdown(
                f"<p style='font-size: 18px;'>Confidence Level: <strong>{confidence:.2%}</strong></p>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"⚠️ Error in prediction: {str(e)}")