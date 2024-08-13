import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv("processed_data.csv")
valid_data = pd.read_csv("valid.csv")

print(train_data.head())

X_train = valid_data

#HANDLING MISSING VALUES
print("No. of Missing Data in Columns")
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print(X_train.shape)

threshold_percentage = 50 
missing_val_count_by_column = X_train.isnull().sum()
percentage_missing = (missing_val_count_by_column / len(X_train)) * 100
columns_to_drop = percentage_missing[percentage_missing > threshold_percentage].index
X_train_filtered = X_train.drop(columns=columns_to_drop)

print("Shape of X_train after removing columns with more than {}% missing values:".format(threshold_percentage))
print(X_train_filtered.shape)

numeric_columns = X_train_filtered.select_dtypes(include='number').columns
categorical_columns = X_train_filtered.select_dtypes(exclude='number').columns

numeric_columns_with_missing_values = [col for col in numeric_columns if X_train_filtered[col].isnull().any()]
categorical_columns_with_missing_values = [col for col in categorical_columns if X_train_filtered[col].isnull().any()]


numeric_imputer = SimpleImputer(strategy='mean') 
categorical_imputer = SimpleImputer(strategy='most_frequent') 

X_train_imputed_numeric = X_train_filtered.copy()
X_train_imputed_numeric[numeric_columns_with_missing_values] = numeric_imputer.fit_transform(X_train_filtered[numeric_columns_with_missing_values])

X_train_imputed_categorical = X_train_imputed_numeric.copy()
X_train_imputed_categorical[categorical_columns_with_missing_values] = categorical_imputer.fit_transform(X_train_imputed_numeric[categorical_columns_with_missing_values])

print(X_train_imputed_categorical.describe())

print("No. of Missing Data in Columns")
missing_val_count_by_column = (X_train_imputed_categorical.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print(X_train_imputed_categorical.shape)

# ENCODING
s = (X_train_imputed_categorical.dtypes == 'object')
object_cols = list(s[s].index)

ordinal_encoder = OrdinalEncoder()
X_train_encoded = X_train_imputed_categorical.copy()
X_train_encoded[object_cols] = ordinal_encoder.fit_transform(X_train_imputed_categorical[object_cols])

print(X_train_encoded.emp_length.head())

#SCALING
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train_encoded)

# Convert the scaled NumPy array back to a DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns)

# View the first few rows of the DataFrame
print(X_train_scaled_df.head())

# Check for columns with zero variance (i.e., all values are the same)
columns_to_drop = X_train_scaled_df.columns[X_train_scaled_df.nunique() == 1]

# Drop the columns with zero variance
X_train_scaled_df.drop(columns=columns_to_drop, inplace=True)
print(X_train_scaled_df.shape)


additional_feature = set(X_train_scaled_df.columns) - set(train_data.columns)
print("Additional feature in valid data:", additional_feature)
X_train_scaled_df = X_train_scaled_df.drop(columns=additional_feature)

# Now both datasets should have the same number of features
print("Train data shape:", train_data.shape)
print("Valid data shape:", X_train_scaled_df.shape)

# Write the DataFrame to a new CSV file
X_train_scaled_df.to_csv('processed_valid.csv', index=False)