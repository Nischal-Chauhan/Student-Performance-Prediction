
import pandas as pd

# Load the Mathematics dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

print("STUDENT DATASET EXPLORATION")
print("--------------------------------")

# Explore final grade
print("\nFinal Grade (G3) Statistics:")
print(df["G3"].describe())

# Count students in each grade
print("\nFinal Grade Distribution:")
print(df["G3"].value_counts().sort_index())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Display categorical columns
categorical_columns = df.select_dtypes(
    include="str"
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

# Display unique values in selected columns
print("\nUnique Values:")
for column in ["school", "sex", "address", "Mjob"]:
    print(f"{column}: {df[column].unique()}")
