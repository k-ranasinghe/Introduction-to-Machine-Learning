import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv("train.csv")
valid_data = pd.read_csv("valid.csv")

print(train_data.head())

X_train = train_data

print("No. of Missing Data in Columns")
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print(X_train.shape)

# Define the threshold percentage
threshold_percentage = 50  # Example threshold: remove columns with more than 50% missing values

# Calculate the number of missing values for each column
missing_val_count_by_column = X_train.isnull().sum()

# Calculate the percentage of missing values for each column
percentage_missing = (missing_val_count_by_column / len(X_train)) * 100

# Identify columns exceeding the threshold percentage
columns_to_drop = percentage_missing[percentage_missing > threshold_percentage].index

# Remove columns exceeding the threshold percentage
X_train_filtered = X_train.drop(columns=columns_to_drop)

# Print the shape of the DataFrame after removing columns
print("Shape of X_train after removing columns with more than {}% missing values:".format(threshold_percentage))
print(X_train_filtered.shape)

# Identify numerical and categorical columns with missing values
numeric_columns = X_train_filtered.select_dtypes(include='number').columns
categorical_columns = X_train_filtered.select_dtypes(exclude='number').columns

numeric_columns_with_missing_values = [col for col in numeric_columns if X_train_filtered[col].isnull().any()]
categorical_columns_with_missing_values = [col for col in categorical_columns if X_train_filtered[col].isnull().any()]

# Choose imputation strategies for numerical and categorical columns
numeric_imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean for numerical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')  # Impute missing values with most frequent category for categorical columns

# Apply imputation to fill missing values for numerical columns
X_train_imputed_numeric = X_train_filtered.copy()
X_train_imputed_numeric[numeric_columns_with_missing_values] = numeric_imputer.fit_transform(X_train_filtered[numeric_columns_with_missing_values])

# Apply imputation to fill missing values for categorical columns
X_train_imputed_categorical = X_train_imputed_numeric.copy()
X_train_imputed_categorical[categorical_columns_with_missing_values] = categorical_imputer.fit_transform(X_train_imputed_numeric[categorical_columns_with_missing_values])

# Validate and evaluate
# For example, you can print summary statistics or visualize distributions to ensure imputation was successful
print(X_train_imputed_categorical.describe())


print("No. of Missing Data in Columns")
missing_val_count_by_column = (X_train_imputed_categorical.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print(X_train_imputed_categorical.shape)


# Plot heatmap of missing values
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.xlabel('Features')
plt.ylabel('Data')
plt.title('Missing Values in Train Dataset')
plt.show()