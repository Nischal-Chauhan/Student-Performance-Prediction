
import pandas as pd

# Load Mathematics dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

print("ZERO GRADE INVESTIGATION")
print("--------------------------------")

# Filter students with final grade zero
zero_students = df[df["G3"] == 0]

print("Students with G3 = 0:", len(zero_students))

print("\nFirst 10 Zero-Grade Records:")

print(
    zero_students[
        ["school", "sex", "age", "studytime",
         "failures", "absences", "G1", "G2", "G3"]
    ].head(10)
)

print("\nAverage Grades of Zero-Grade Students:")

print(
    zero_students[["G1", "G2", "G3", "absences"]].mean()
)

print("\nAverage Grades of Other Students:")

other_students = df[df["G3"] > 0]

print(
    other_students[["G1", "G2", "G3", "absences"]].mean()
)

print("\nZero-Grade Investigation Completed.")
