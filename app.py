import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set up page layout and styling
st.set_page_config(page_title="Interactive Iris Analytics", layout="wide")

st.title("🌸 Interactive Iris Flower Classification Dashboard")
st.write("Experiment with feature dimensions and model hyperparameters in real-time to see how machine learning boundaries shift.")

# --- 1. Sidebar: Model Hyperparameter Tuning ---
st.sidebar.header("⚙️ Model Hyperparameters")

st.sidebar.subheader("K-Nearest Neighbors")
k_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)

st.sidebar.subheader("Random Forest")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, step=10)

# --- 2. Sidebar: Feature Input Sliders ---
st.sidebar.markdown("---")
st.sidebar.header("📏 Input Flower Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 6.9, 4.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# --- 3. Dynamic Model Training ---
@st.cache_resource
def get_base_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    return X_train, X_test, y_train, y_test, iris.target_names, iris.feature_names

X_train, X_test, y_train, y_test, target_names, feature_names = get_base_data()

# Scale features dynamically based on training split
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train models with user-selected sidebar settings
knn_model = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_train_scaled, y_train)
dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(X_train, y_train)

# --- 4. Process Live Predictions ---
user_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
user_features_scaled = scaler.transform(user_features)

predictions = {
    "KNN": {
        "pred": knn_model.predict(user_features_scaled)[0],
        "prob": knn_model.predict_proba(user_features_scaled)[0]
    },
    "Decision Tree": {
        "pred": dt_model.predict(user_features)[0],
        "prob": dt_model.predict_proba(user_features)[0]
    },
    "Random Forest": {
        "pred": rf_model.predict(user_features)[0],
        "prob": rf_model.predict_proba(user_features)[0]
    }
}

# --- 5. UI Layout: Real-Time Prediction Cards ---
st.subheader("🔮 Live Model Predictions")
cols = st.columns(3)

# Distinct styling colors for flower classes
class_colors = {"setosa": "🔴", "versicolor": "🟢", "virginica": "🔵"}

for i, (name, data) in enumerate(predictions.items()):
    with cols[i]:
        pred_species = target_names[data["pred"]]
        confidence = data["prob"][data["pred"]] * 100
        
        st.markdown(f"### {name}")
        st.metric(
            label="Predicted Species", 
            value=f"{class_colors[pred_species]} {pred_species.title()}"
        )
        st.progress(int(confidence))
        st.caption(f"Confidence: **{confidence:.1f}%**")

# --- 6. UI Layout: Interactive Probability Breakdown ---
st.markdown("---")
st.subheader("📊 Class Probability Distribution Comparison")

# Build a DataFrame for clean visualization plotting
prob_data = []
for name, data in predictions.items():
    for target_idx, prob_val in enumerate(data["prob"]):
        prob_data.append({
            "Model": name,
            "Species": target_names[target_idx].title(),
            "Probability": prob_val
        })
prob_df = pd.DataFrame(prob_data)

# Render native Streamlit interactive bar chart
st.bar_chart(
    prob_df, 
    x="Species", 
    y="Probability", 
    color="Model", 
    stack=False
)

# --- 7. UI Layout: Feature Context Guide ---
st.markdown("---")
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📋 Your Custom Input")
    input_df = pd.DataFrame(user_features, columns=feature_names)
    st.dataframe(input_df, hide_index=True)

with col_right:
    st.subheader("💡 What do these metrics mean?")
    st.markdown(
        """
        - **Sepal vs Petal:** Sepals are the outer green leaves protecting the bud; petals are the bright colorful parts of the flower.
        - **Why do models change configurations?** Try changing the **Number of Neighbors (K)** in the sidebar to `1`. 
          Notice how the KNN confidence jumps immediately to 100% or switches radically. Increasing K smooths out these abrupt shifts.
        """
    )