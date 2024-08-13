import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Load data
train_data = pd.read_csv("train.csv")
valid_data = pd.read_csv("valid.csv")

print(train_data.head())

X_train = train_data.drop(columns=['loan_status'])
y_train = train_data['loan_status']

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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns)
print(X_train_scaled_df.head())

columns_to_drop = X_train_scaled_df.columns[X_train_scaled_df.nunique() == 1]
X_train_scaled_df.drop(columns=columns_to_drop, inplace=True)
print(X_train_scaled_df.shape)


#SELECTION
# Initialize the classifier
clf = RandomForestClassifier()

# Define a range of k values (number of features to select)
k_values = [10]

for k in k_values:
    # Initialize RFE with the classifier and the number of features to select
    selector = RFE(estimator=clf, n_features_to_select=k, verbose=2) 

    # Fit RFE to the training data
    selector.fit(X_train_scaled_df, y_train)

    # Get selected features
    selected_features = X_train_scaled_df.columns[selector.support_]

    # Evaluate the model using cross-validation 
    scores = cross_val_score(clf, selector.transform(X_train_scaled_df), y_train, cv=5, scoring='accuracy')

    # Output selected features and performance score
    print(f"Number of features selected: {k}, Selected features: {selected_features}, Performance score: {np.mean(scores):.4f}")