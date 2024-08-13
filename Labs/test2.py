import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import shap

# Load data
train_data = pd.read_csv("ML.csv")
valid_data = pd.read_csv("ML1.csv")
train_data1 = pd.read_csv("train.csv")
valid_data1 = pd.read_csv("valid.csv")

features = ['loan_amnt', 'term', 'int_rate', 'home_ownership', 'annual_inc', 'verification_status', 'earliest_cr_line', 'inq_last_6mths', 'revol_util', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'tot_cur_bal', 'acc_open_past_24mths', 'bc_open_to_buy', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'num_actv_bc_tl', 'debt_settlement_flag', 'issue_y']

X_train = train_data[features]
y_train = train_data1['loan_status']
X_valid = valid_data[features]
y_valid = valid_data1['loan_status']

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
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the validation set
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(X_valid, y_valid)
print("Validation Accuracy:", accuracy)

xgb_model = XGBClassifier()
y_valid_pred = best_clf.predict(X_valid)
accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"Accuracy: {accuracy:.2f}")


# Train XGBoost model with the best hyperparameters
best_clf.fit(X_train, y_train)

# Initialize the explainer with the trained model
explainer = shap.Explainer(best_clf)

# Calculate Shapley values for the validation set
shap_values = explainer.shap_values(X_valid)

# Visualize Shapley values
shap.summary_plot(shap_values, X_valid)