import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

train_data = pd.read_csv("ML.csv")
train_data1 = pd.read_csv("train.csv")

X_train = train_data.drop(columns=['loan_status'])
y_train = train_data1['loan_status']

#SELECTION
# Initialize the classifier
clf = RandomForestClassifier()

# Define a range of k values (number of features to select)
k_values = [15]

for k in k_values:
    # Initialize RFE with the classifier and the number of features to select
    selector = RFE(estimator=clf, n_features_to_select=k, verbose=2) 

    # Fit RFE to the training data
    selector.fit(X_train, y_train)

    # Get selected features
    selected_features = X_train.columns[selector.support_]

    # Evaluate the model using cross-validation 
    scores = cross_val_score(clf, selector.transform(X_train), y_train, cv=5, scoring='accuracy')

    # Output selected features and performance score
    print(f"Number of features selected: {k}, Selected features: {selected_features}, Performance score: {np.mean(scores):.4f}")
