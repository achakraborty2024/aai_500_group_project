import numpy as np
import pandas as pd

file_path = 'VM.csv'

filtered_file_path = file_path.split('.')[0] + '_filtered.csv'
df = pd.read_csv(file_path)

# Delete the column
df.drop('Time', axis=1, inplace=True)

df.to_csv(file_path, index=False)

df = pd.read_csv(file_path)

# # Sample data
data = df['Execution Time']

# Calculate Q1, Q3, and IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - data

# Determine bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Outliers: {outliers}")

filtered_df = data[(data >= lower_bound) | (data <= upper_bound)]

# # Save the filtered DataFrame back to a new CSV file, without the unwanted rows
filtered_df.to_csv(filtered_file_path, index=False)
