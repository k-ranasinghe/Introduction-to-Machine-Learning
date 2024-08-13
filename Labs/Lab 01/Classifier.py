import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Load data
train_data = pd.read_csv("train.csv")
valid_data = pd.read_csv("valid.csv")

X_train = train_data
X_valid = valid_data.drop(columns=['loan_status'])
y_valid = valid_data['loan_status']

threshold_percentage = 50 
missing_val_count_by_column = X_train.isnull().sum()
percentage_missing = (missing_val_count_by_column / len(X_train)) * 100
columns_to_drop = percentage_missing[percentage_missing > threshold_percentage].index
X_train_filtered = X_train.drop(columns=columns_to_drop)

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

# ENCODING
s = (X_train_imputed_categorical.dtypes == 'object')
object_cols = list(s[s].index)

ordinal_encoder = OrdinalEncoder()
X_train_encoded = X_train_imputed_categorical.copy()
X_train_encoded[object_cols] = ordinal_encoder.fit_transform(X_train_imputed_categorical[object_cols])

# Check for columns with zero variance (i.e., all values are the same)
columns_to_drop = X_train_encoded.columns[X_train_encoded.nunique() == 1]

# Drop the columns with zero variance
X_train_encoded.drop(columns=columns_to_drop, inplace=True)


# Split dataset into features (X) and target variable (y)
X = X_train_encoded.drop(columns=['loan_status'])  # Adjust 'target_column' with your target variable name
y = X_train_encoded['loan_status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Initialize the classifier
# clf = xgb.XGBClassifier(tree_method='hist', device = "cuda")

# # Train the classifier
# clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss', verbose=True)

# Initialize the classifier
clf = XGBClassifier()

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the validation set
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(X_valid, y_valid)
print("Validation Accuracy:", accuracy)
