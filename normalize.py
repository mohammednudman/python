import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sample dataset (replace with your own data)
data = np.array([[1.0, 2.0],
                 [3.0, 4.0],
                 [5.0, 6.0]])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to your data and transform it
normalized_data = scaler.fit_transform(data)

# Print the normalized data
print("Normalized Data:")
print(normalized_data)