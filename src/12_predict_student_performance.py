
import pandas as pd
import joblib

# Load saved model
model = joblib.load("models/random_forest_model.pkl")

# Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Select test records
X = df.drop(columns=["G1", "G2", "G3"])

# Take first 5 records as demonstration inputs
sample_students = X.head(5)

# Predict final grades
predictions = model.predict(sample_students)

# Display predictions
print("STUDENT PERFORMANCE PREDICTIONS")
print("----------------------------------------")

for i, prediction in enumerate(predictions):
    print(
        f"Student {i + 1}: Predicted Final Grade = "
        f"{prediction:.2f}/20"
    )

print("\nPredictions completed successfully.")
