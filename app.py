import os

import streamlit as st
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title="Breast Cancer KNN System",
    page_icon="🎗️",
    layout="wide"
)

# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "knn_sklearn.pkl"))

scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

features = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))
# CSS
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3, p {
    color: white;
}

.hero {
    padding: 70px;
    border-radius: 25px;
    background: linear-gradient(135deg,#9333ea,#ec4899);
    text-align: center;
    margin-bottom: 40px;
}

.hero-title {
    color: white;
    font-size: 55px;
    font-weight: bold;
}

.hero-sub {
    color: white;
    font-size: 22px;
    margin-top: 15px;
}

/* 🔥 UPDATED CARDS (same gradient as hero) */
.card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(135deg,#9333ea,#ec4899);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.4);
    transition: 0.3s;
    color: white;
}

.card:hover {
    transform: scale(1.05);
}

.card-title {
    font-size: 24px;
    font-weight: bold;
    color: white;
}

.card-text {
    color: #fce7f3;
    margin-top: 10px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: #111827;
    color: white;
    font-weight: bold;
    border: 1px solid #9333ea;
}

.stButton > button:hover {
    background: #1f2937;
    transform: scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# HERO SECTION
st.markdown("""
<div class='hero'>
    <div class='hero-title'>🎗️ Breast Cancer Prediction System</div>
    <div class='hero-sub'>Machine Learning Project using KNN Algorithm</div>
</div>
""", unsafe_allow_html=True)

# CARDS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>📊 Dashboard</div>
        <div class='card-text'>Dataset insights, PCA, ROC & evaluation</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Open Dashboard"):
        st.switch_page("pages/dashboard.py")

with col2:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>🩺 Prediction</div>
        <div class='card-text'>Predict tumor type using KNN model</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Prediction"):
        st.switch_page("pages/prediction.py")

with col3:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>📘 About KNN</div>
        <div class='card-text'>Learn how KNN works interactively</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Learn KNN"):
        st.switch_page("pages/about.py")