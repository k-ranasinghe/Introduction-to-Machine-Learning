# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import xgboost as xgb
import shap

# Load data
train_data = pd.read_csv("train.csv")
valid_data = pd.read_csv("valid.csv")

# Separate features and target variable
X_train = train_data.drop(columns=['loan_status'])
y_train = train_data['loan_status']
X_valid = valid_data.drop(columns=['loan_status'])
y_valid = valid_data['loan_status']

# Preprocessing pipeline
categorical_features = X_train.select_dtypes(include=['object']).columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# XGBoost Classifier with feature selection
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
    ('classifier', xgb.XGBClassifier())
])

# Fit the model
xgb_model.fit(X_train, y_train)

# Evaluate model
accuracy = xgb_model.score(X_valid, y_valid)
print("Validation Accuracy:", accuracy)

# Dimensionality Reduction using PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)

# XGBoost Classifier with PCA
xgb_model_pca = xgb.XGBClassifier()
xgb_model_pca.fit(X_train_pca, y_train)
accuracy_pca = xgb_model_pca.score(X_valid_pca, y_valid)
print("Validation Accuracy after PCA:", accuracy_pca)

# Get reduced set of features
selected_features = X_train.columns[xgb_model.named_steps['feature_selection'].get_support()]
print("Selected Features:", selected_features)

# Explain the model using Shapely values
explainer = shap.Explainer(xgb_model.named_steps['classifier'], X_train)
shap_values = explainer(X_valid)

# Visualize the Shapely values
shap.summary_plot(shap_values, X_valid)
