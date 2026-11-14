import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Separate features and target
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

# Identify feature types
numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Create pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Hyperparameter combinations
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 5, 10],
    "regressor__min_samples_split": [2, 5],
    "regressor__min_samples_leaf": [1, 2]
}

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("RANDOM FOREST - HYPERPARAMETER TUNING")
print("----------------------------------------")
print("Training models. Please wait...")

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "models/random_forest_model.pkl")

print("\nTrained model saved successfully!")

# Predictions
predictions = best_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nBEST PARAMETERS")
print("----------------------------------------")
print(grid_search.best_params_)

print("\nTUNED MODEL PERFORMANCE")
print("----------------------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

print("\nHyperparameter tuning completed successfully.")
