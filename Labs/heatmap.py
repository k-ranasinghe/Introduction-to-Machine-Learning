import pandas as pd

# Load data
test_data = pd.read_csv("X_test.csv")
valid_data = pd.read_csv("valid.csv")

print(test_data.shape)
print(valid_data.shape)