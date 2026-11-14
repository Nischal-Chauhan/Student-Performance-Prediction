
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Create reports folder output paths
print("STUDENT PERFORMANCE DATA VISUALIZATION")
print("----------------------------------------")

# 1. Final grade distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=20, kde=True)

plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig("reports/grade_distribution.png")
plt.show()

# 2. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()

plt.savefig("reports/studytime_vs_grade.png")
plt.show()

# 3. Previous grades vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="G1", y="G3")

plt.title("First Period Grade vs Final Grade")
plt.xlabel("First Period Grade (G1)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()

plt.savefig("reports/g1_vs_g3.png")
plt.show()

print("\nVisualizations saved successfully in reports/ folder.")
