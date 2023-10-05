# %%
!pip install numpy scikit-learn matplotlib

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# %%
np.random.seed(42)
X = 0.3 * np.random.randn(100, 2)
X = np.vstack([X, 2 * np.random.randn(10, 2)])

# %%
plt.scatter(X[:, 0], X[:, 1], s=100, edgecolors='k', alpha=0.5)
plt.title("Input Data")
plt.show()

# %%
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# %%
y_pred = clf.predict(X)

# %%
plt.scatter(X[:, 0], X[:, 1], s=100, edgecolors='k', c=y_pred, cmap=plt.cm.coolwarm)
plt.title("Isolation Forest Outlier Detection")
plt.show()

# %%
num_outliers = np.sum(y_pred == -1)
print(f"Number of outliers detected: {num_outliers}")

# %%



