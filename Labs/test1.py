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

# Initialize SelectKBest with the scoring function and the number of features to select
selector = SelectKBest(score_func=f_classif)

# Define a range of k values (number of features to select)
k_values = [5, 10, 15, 20, 25]

for k in k_values:
    print(f"Selecting top {k} features...")
    # Set the number of features to select
    selector.k = k

    # Fit selector to the training data
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Get the indices of selected features
    selected_indices = selector.get_support(indices=True)

    # Get the names of selected features
    selected_features = X_train.columns[selected_indices]

    # Evaluate the model using cross-validation
    scores = cross_val_score(clf, X_train_selected, y_train, cv=5, scoring='accuracy')

    # Output selected features and performance score
    print(f"Number of features selected: {k}, Selected features: {selected_features}, Performance score: {np.mean(scores):.4f}")
