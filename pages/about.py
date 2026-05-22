import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# PAGE CONFIG
st.set_page_config(layout="wide")

st.title("🤖 About K-Nearest Neighbors (KNN)")

st.markdown("---")

# INTRODUCTION
st.markdown("""
#  What is KNN?

K-Nearest Neighbors (KNN) is a **supervised machine learning algorithm**
used for classification problems.

It is a **lazy learning algorithm**, meaning it stores data instead of learning a model.
""")

st.markdown("---")

# HOW IT WORKS
st.markdown("""
# ⚙️ How KNN Works

1. Choose K value  
2. Calculate distance between points  
3. Find K nearest neighbors  
4. Majority voting decides the class  
""")

st.markdown("---")

# DISTANCE
st.markdown("""
# 📏 Distance Calculation

KNN uses distance metrics like:

- Euclidean Distance (most common)
- Manhattan Distance

\[
d = \sqrt{\sum (x_i - y_i)^2}
\]
""")

st.markdown("---")

# WHY SCALING
st.markdown("""
# ⚖️ Why Feature Scaling is Important?

- KNN depends on distance  
- Large values dominate  
- Scaling makes features fair  

✔ StandardScaler is required
""")

st.markdown("---")

# ROLE OF K
st.markdown("""
# 🎯 Choosing K

- Small K → noisy (overfitting)
- Large K → smooth (underfitting)
""")

# 🚀 INTERACTIVE PART 

st.markdown("---")
st.header("📊 Interactive KNN Demo ")

# small dataset for speed
X, y = make_classification(
    n_samples=120,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42
)

# SIDEBAR CONTROLS
st.sidebar.header("⚙️ Controls")

k = st.sidebar.slider("K Value", 1, 15, 3)

x1 = st.sidebar.slider("Feature 1", float(X[:,0].min()), float(X[:,0].max()), 0.0)
x2 = st.sidebar.slider("Feature 2", float(X[:,1].min()), float(X[:,1].max()), 0.0)

# MODEL
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X, y)

pred = model.predict([[x1, x2]])[0]

# PLOT
fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.7)
ax.scatter(x1, x2, c="black", s=150, marker="X")

ax.set_title(f"KNN Visualization (K={k})")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

st.pyplot(fig)

# RESULT
st.markdown("---")

if pred == 1:
    st.success("🟢 Prediction: Class 1 (Benign)")
else:
    st.error("🔴 Prediction: Class 0 (Malignant)")

st.info("Prediction based on majority vote of nearest neighbors")