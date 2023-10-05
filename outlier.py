import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors

# Generate synthetic data
np.random.seed(42)
X = 0.3 * np.random.randn(100, 2)
X = np.vstack([X, 2 * np.random.randn(10, 2)])

# Calculate angle-based outlier scores
def angle_based_outlier_score(X, k=10):
    # (Same as before, omitting for brevity)
    # ...
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    mean_distance = np.mean(distances[:, 1:], axis=1)
    angles = np.arccos(1 - (mean_distance / np.max(mean_distance)))
    return angles


angles = angle_based_outlier_score(X)

# Set a threshold to classify inliers and outliers
threshold = 1.0

# Define true labels based on your domain knowledge
# You need to manually label your data as inliers and outliers
# For this example, assuming that the first 100 points are inliers and the last 10 points are outliers
true_labels = np.array([1] * 100 + [0] * 10)

# Classify data based on the threshold
predicted_labels = (angles > threshold).astype(int)

# Create a confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Calculate accuracy (optional)
accuracy = accuracy_score(true_labels, predicted_labels)

print("Confusion Matrix:")
print(confusion)
print(f"Accuracy: {accuracy:.2f}")
