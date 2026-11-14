
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Prepare features and target
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build preprocessing and model pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Actual vs predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions)

plt.plot([0, 20], [0, 20], linestyle="--")

plt.title("Actual vs Predicted Student Grades")
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")

plt.xlim(0, 20)
plt.ylim(0, 20)

plt.tight_layout()
plt.savefig("reports/actual_vs_predicted.png")
plt.show()

print("Prediction visualization saved successfully.")
