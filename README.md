# Student Performance Prediction

A Machine Learning project that predicts students' final Mathematics grades using demographic, family, social, and study-related information.

## Project Overview

The objective of this project is to build a regression model that predicts a student's final Mathematics grade (G3), on a scale of 0–20.

This project follows an early-prediction approach: the first-period (G1) and second-period (G2) grades are excluded from the model inputs.

## Dataset

- **Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Subject:** Mathematics
- **Records:** 395
- **Columns:** 33
- **Target:** G3 (final grade)
- **License:** CC BY 4.0

The original dataset is stored in `data/raw/`.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Project Structure

```text
Student-Performance-Prediction/
├── data/
│   └── raw/
├── models/
│   └── random_forest_model.pkl
├── notebooks/
├── reports/
│   ├── grade_distribution.png
│   ├── studytime_vs_grade.png
│   └── g1_vs_g3.png
├── src/
│   ├── 01_inspect_dataset.py
│   ├── 02_explore_dataset.py
│   ├── 03_visualize_data.py
│   ├── 04_investigate_zero_grades.py
│   ├── 05_feature_relationships.py
│   ├── 06_prepare_features.py
│   ├── 07_preprocess_data.py
│   ├── 08_train_baseline_model.py
│   ├── 09_visualize_predictions.py
│   ├── 10_compare_models.py
│   ├── 11_tune_random_forest.py
│   └── 12_predict_student_performance.py
├── .gitignore
├── LICENSE
└── README.md
```

## Project Workflow

1. Dataset inspection
2. Exploratory data analysis
3. Data visualization
4. Investigation of zero final grades
5. Feature relationship analysis
6. Feature selection
7. Data preprocessing
8. Baseline model training
9. Prediction visualization
10. Machine learning model comparison
11. Random Forest hyperparameter tuning
12. Model saving and prediction

## Models Compared

Four regression algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 3.00 | 3.77 | 0.307 |
| Gradient Boosting | 3.11 | 3.93 | 0.247 |
| Linear Regression | 3.40 | 4.20 | 0.141 |
| Decision Tree | 3.63 | 4.84 | -0.141 |

### Best Model: Tuned Random Forest

The Random Forest model was further optimized using GridSearchCV with 5-fold cross-validation.

Best parameters:

- `n_estimators`: 200
- `max_depth`: None
- `min_samples_split`: 5
- `min_samples_leaf`: 1

Final test-set performance:

| Metric | Result |
|---|---:|
| MAE | 2.99 |
| RMSE | 3.76 |
| R² Score | 0.312 |

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Nischal-Chauhan/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 5. Run the prediction script

```bash
python src/12_predict_student_performance.py
```

The script loads the saved Random Forest model and generates final-grade predictions for sample student records.

## Limitations

- The dataset contains 395 Mathematics student records.
- The model uses demographic, family, social, and study-related features.
- G1 and G2 are excluded from the early-prediction model.
- The tuned model achieved an R² score of 0.312, so its predictive performance remains limited.
- Predictions should not be treated as guaranteed academic outcomes.

## Author

**Nischal Chauhan**

GitHub: [Nischal-Chauhan](https://github.com/Nischal-Chauhan)
