import os
import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer

# LOAD MODELS 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "knn_sklearn.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

# DATA (for reference)
data = load_breast_cancer()
X = np.array(data.data)
y = np.array(data.target)

# PAGE CONFIG
st.set_page_config(
    page_title="Tumor Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System")
st.write("Use sliders or presets 👇")

# PRESETS
benign_sample = [
    0.05, 90.0, 0.10, 0.05, 0.10,
    0.25, 0.25, 15.0, 0.10, 0.005
]

malignant_sample = X[y == 0].mean(axis=0)[:len(features)].tolist()

colA, colB = st.columns(2)

if colA.button("🟢 Load Benign"):
    st.session_state["preset"] = benign_sample

if colB.button("🔴 Load Malignant"):
    st.session_state["preset"] = malignant_sample

if "preset" not in st.session_state:
    st.session_state["preset"] = [0.1] * len(features)

# INPUT SLIDERS (TWO COLUMNS)
st.write("---")
st.subheader("🎛️ Input Features")

input_data = []

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    default = float(st.session_state["preset"][i])

    if i % 2 == 0:
        with col1:
            val = st.slider(
                feature,
                min_value=0.0,
                max_value=200.0 if "perimeter" in feature or "radius" in feature else 50.0,
                value=default
            )
            input_data.append(val)
    else:
        with col2:
            val = st.slider(
                feature,
                min_value=0.0,
                max_value=200.0 if "perimeter" in feature or "radius" in feature else 50.0,
                value=default
            )
            input_data.append(val)

# save state
st.session_state["preset"] = input_data

# PREDICTION
st.write("---")

if st.button("🔍 Predict"):

    x = np.array([input_data])
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)
    prob = model.predict_proba(x_scaled)
    if np.max(prob) < 0.6:
       st.warning("⚠️ Low confidence prediction (model is unsure)")
    
    # RESULT

    if pred[0] == 1:
        st.success("✅ Prediction: Benign Tumor")
    else:
        st.error("⚠️ Prediction: Malignant Tumor")

    # Confidence UI
    st.progress(float(np.max(prob)))
    st.info(f"Confidence: {np.max(prob)*100:.2f}%")

    # DEBUG SECTION (IMPORTANT FOR DISCUSSION)
    with st.expander("🔍 Debug Information", expanded=False):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Raw Input Values")
            st.write(pd.DataFrame([input_data], columns=features))

        with col2:
            st.subheader("⚙️ Scaled Input")
            st.write(pd.DataFrame(x_scaled, columns=features))

        st.subheader(" Feature Order (VERY IMPORTANT)")
        st.write(features)

        st.subheader("📊 Prediction Probabilities")
        st.write({
            "Malignant (0)": float(prob[0][0]),
            "Benign (1)": float(prob[0][1])
        })