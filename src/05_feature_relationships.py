
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

print("FEATURE RELATIONSHIP ANALYSIS")
print("--------------------------------")

# Calculate correlations between numerical columns
correlations = df.corr(numeric_only=True)

# Get correlations with final grade
grade_correlations = correlations["G3"].sort_values(
    ascending=False
)

print("\nCorrelation with Final Grade (G3):")
print(grade_correlations)

print("\nTop 5 Positively Related Features:")
print(
    grade_correlations.drop("G3").head(5)
)

print("\nTop 5 Negatively Related Features:")
print(
    grade_correlations.drop("G3").tail(5)
)

print("\nFeature Relationship Analysis Completed.")
