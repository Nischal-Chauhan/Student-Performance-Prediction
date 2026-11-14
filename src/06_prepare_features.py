
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

print("FEATURE PREPARATION REPORT")
print("--------------------------------")

# Target: final grade
y = df["G3"]

# Early prediction: exclude G1 and G2
X = df.drop(columns=["G1", "G2", "G3"])

# Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("Total Records:", len(df))
print("Number of Input Features:", X.shape[1])

print("\nInput Dataset Shape:", X.shape)
print("Target Dataset Shape:", y.shape)

print("\nNumerical Features:", len(numerical_features))
print(numerical_features)

print("\nCategorical Features:", len(categorical_features))
print(categorical_features)

print("\nTarget (G3) Distribution:")
print(y.describe())

print("\nFeature Preparation Completed.")
