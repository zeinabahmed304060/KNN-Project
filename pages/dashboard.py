import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

# PAGE CONFIG
st.set_page_config(
    page_title="KNN AI Dashboard",
    layout="wide",
    page_icon="🧠"
)

# BASE PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LOAD REPORT
def load_json(name):
    path = os.path.join(BASE_DIR, "reports", name)
    with open(path) as f:
        return json.load(f)

scratch_report = load_json("report_scratch.json")
sklearn_report = load_json("report_sklearn.json")

# BEST K (FIXED PATH)
best_k_path = os.path.join(BASE_DIR, "reports", "best_k.json")

with open(best_k_path) as f:
    best_k_data = json.load(f)

best_k = best_k_data["best_k"]
error_rate = best_k_data["error_rate"]
# CLEAN REPORT
def to_df(report):
    df = pd.DataFrame(report).T
    df = df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    return df

scratch_df = to_df(scratch_report)
sklearn_df = to_df(sklearn_report)
# DATASET
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

features = X.columns[:10]
X_sel = X[features]

# MODEL (Scaling + Probability) for ROC
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)
y_proba = knn.predict_proba(X_scaled)[:, 1]

# CSS
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1,h2,h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)
# TITLE
st.title("🧠 KNN Models Dashboard")
st.write("Scratch vs Sklearn KNN Comparison")
# TAB
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📂 Dataset Insights", "📈 Evaluation"])

# TAB 1 - OVERVIEW
with tab1:

    st.subheader("📊 Project Summary")
    st.info("""
    This project compares KNN implemented from scratch vs Scikit-Learn KNN
    for Breast Cancer classification using medical diagnostic features.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Scratch Accuracy", f"{scratch_report['accuracy']*100:.2f}%")
    col2.metric("Sklearn Accuracy", f"{sklearn_report['accuracy']*100:.2f}%")

    best = "Scratch" if scratch_report['accuracy'] > sklearn_report['accuracy'] else "Sklearn"
    col3.metric("Best Model", best)

    comp = pd.DataFrame({
        "Model": ["Scratch", "Sklearn"],
        "Accuracy": [scratch_report['accuracy'], sklearn_report['accuracy']]
    })

    fig = px.bar(comp, x="Model", y="Accuracy", text_auto=".2%")
    st.plotly_chart(fig, use_container_width=True)

    # BEST K ANALYSIS
    st.subheader("📈 Best K Analysis (Error Rate vs K)")

    fig_k = px.line(
        x=list(range(1, len(error_rate) + 1)),
        y=error_rate,
        markers=True
    )

    fig_k.add_vline(x=best_k, line_dash="dash", line_color="red")

    st.metric("Best K Value", best_k)
    st.plotly_chart(fig_k, use_container_width=True)

# TAB 2 - DATASET INSIGHTS
with tab2:

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", len(X))
    col2.metric("Features", X.shape[1])
    col3.metric("Classes", "Benign / Malignant")

    # CLASS DISTRIBUTION
    st.subheader("📊 Class Distribution")

    dist = pd.DataFrame({"Class": y})
    dist["Class"] = dist["Class"].map({0: "Malignant", 1: "Benign"})

    fig = px.pie(dist, names="Class")
    st.plotly_chart(fig, use_container_width=True)

    class_counts = dist["Class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]

    fig2 = px.bar(class_counts, x="Class", y="Count", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

    # CORRELATION MATRIX
    st.subheader("🔥 Correlation Matrix (Top 10 Features)")

    corr_features = X[features[:]]

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(
        corr_features.corr(),
        cmap="coolwarm",
        annot=True,
        linewidths=0.5,
        ax=ax,
        
    )
    st.pyplot(fig)

    # FEATURE IMPORTANCE
    st.subheader("📊 Feature Importance (Correlation with Target)")

    corr_imp = X_sel.corrwith(pd.Series(y)).abs().sort_values(ascending=False)

    fig3 = px.bar(
        x=corr_imp.index,
        y=corr_imp.values
    )
    st.plotly_chart(fig3, use_container_width=True)

    # PCA
    st.subheader("📉 PCA Visualization")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_sel))

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Class"] = y

    fig4 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=pca_df["Class"].map({0: "Malignant", 1: "Benign"})
    )

    st.plotly_chart(fig4, use_container_width=True)

# TAB 3 - EVALUATION
with tab3:

    st.subheader("📋 Classification Reports")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Scratch Model")
        st.dataframe(scratch_df)

    with c2:
        st.write("Sklearn Model")
        st.dataframe(sklearn_df)

    st.markdown("---")

    st.subheader("🔥 Metric Comparison")

    metric = st.selectbox("Choose Metric", ["precision", "recall", "f1-score"])

    compare = pd.DataFrame({
        "Class": scratch_df.index,
        "Scratch": scratch_df[metric].values,
        "Sklearn": sklearn_df[metric].values
    })

    fig5 = px.bar(compare, x="Class", y=["Scratch", "Sklearn"], barmode="group")
    st.plotly_chart(fig5, use_container_width=True)

    # CONFUSION MATRIX
    st.subheader("🧩 Confusion Matrix")

    img_path = os.path.join(BASE_DIR, "assets", "confusion_matrices.png")

    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning("Confusion matrix image not found")

    # ROC + AUC
    st.subheader("📈 ROC Curve + AUC")

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    fig7 = px.line(x=fpr, y=tpr)

    fig7.add_shape(
        type="line",
        x0=0, x1=1, y0=0, y1=1,
        line=dict(dash="dash")
    )

    st.metric("AUC Score", f"{roc_auc:.3f}")
    st.plotly_chart(fig7, use_container_width=True)