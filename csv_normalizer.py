import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the CSV data into a DataFrame
file_path = 'data.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Specify the name of the column to normalize
column_name = 'column_to_normalize'  # Replace with the actual column name

# Convert the specified column from strings to numerical values using label encoding
label_encoder = LabelEncoder()
df[column_name] = label_encoder.fit_transform(df[column_name])

# Normalize the numerical values in the specified column
scaler = MinMaxScaler()
df[column_name] = scaler.fit_transform(df[[column_name]])

# Save the modified DataFrame back to a CSV file (optional)
output_file_path = 'normalized_data.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, index=False)
