import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import shap

# Load data
train_data = pd.read_csv("processed_data.csv")
valid_data = pd.read_csv("processed_valid.csv")
train_data1 = pd.read_csv("train.csv")
valid_data1 = pd.read_csv("valid.csv")

# features = ['int_rate', 'grade', 'sub_grade', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'debt_settlement_flag']
features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt']

X_train = train_data[features]
y_train = train_data1['loan_status']
X_valid = valid_data[features]
y_valid = valid_data1['loan_status']


# Step 2: Initialize PCA and fit to the standardized data
pca = PCA(n_components=0.95)  # Retain 95% of the variance

X_train_pca = pca.fit_transform(X_train)
X_train_pca_df = pd.DataFrame(X_train_pca)

X_valid_pca = pca.fit_transform(X_valid)
X_valid_pca_df = pd.DataFrame(X_valid_pca)

# Now both datasets should have the same number of features
print("Train data shape:", X_train_pca_df.shape)
print("Valid data shape:", X_valid_pca_df.shape)


# Initialize the classifier
clf = xgb.XGBClassifier(tree_method='hist', device = "cuda")

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [7],
    'learning_rate': [0.1],
    'n_estimators': [300],
    'gamma': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X_train_pca_df, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the validation set
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(X_valid_pca_df, y_valid)
print("Validation Accuracy:", accuracy)







# Train XGBoost model with the best hyperparameters
best_clf.fit(X_train_pca_df, y_train)

# Initialize the explainer with the trained model
explainer = shap.Explainer(best_clf)

# Calculate Shapley values for the validation set
shap_values = explainer.shap_values(X_valid_pca_df)

# Visualize Shapley values
shap.summary_plot(shap_values, X_valid_pca_df)

# Visualize individual SHAP explanations for a specific instance
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_valid_pca_df.iloc[0, :])

# Visualize dependence plot for a specific feature
shap.dependence_plot("1", shap_values, X_valid)