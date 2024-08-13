import pandas as pd
from sklearn.decomposition import PCA

# Load data
train_data = pd.read_csv("processed_data.csv")
valid_data = pd.read_csv("processed_valid.csv")
train_data1 = pd.read_csv("train.csv")
valid_data1 = pd.read_csv("valid.csv")

X_train = train_data.drop(columns=['loan_status'])
y_valid = train_data1['loan_status']
X_valid = valid_data.drop(columns=['loan_status'])
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