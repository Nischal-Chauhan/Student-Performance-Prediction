
import pandas as pd

# Load the Mathematics dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Display basic information
print("STUDENT PERFORMANCE DATASET")
print("--------------------------------")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())
