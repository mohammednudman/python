# %% [markdown]
# ISSE GLOBALLY INSTALL KAR

# %%
!pip install streamlit

# %% [markdown]
# Ek py file me yeh bana

# %%
import streamlit as st
import numpy as np
from sklearn.ensemble import IsolationForest

st.title("Isolation Forest Outlier Detection App")

# Function to detect outliers
def detect_outliers(X, contamination):
    clf = IsolationForest(contamination=contamination, random_state=42)
    y_pred = clf.fit_predict(X)
    return y_pred

# Create a sidebar for setting the contamination parameter
st.sidebar.header("Settings")
contamination = st.sidebar.slider("Contamination", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

# Generate synthetic data (you can replace this with your own data)
np.random.seed(42)
X = 0.3 * np.random.randn(100, 2)
X = np.vstack([X, 2 * np.random.randn(10, 2)])

# Detect outliers
y_pred = detect_outliers(X, contamination)

# Display the number of outliers detected
num_outliers = np.sum(y_pred == -1)
st.write(f"Number of outliers detected: {num_outliers}")

# Plot the data with outliers highlighted
st.scatter_chart(X, c=y_pred, use_container_width=True)

# %% [markdown]
# Yeh command use karke run kar

# %%
streamlit run outlier_detection_app.py


