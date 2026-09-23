# ============================================================
# ADULT INCOME - LOGISTIC REGRESSION
# Binary Income Classification
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_FILE = "adult/adult.csv"
MODEL_FILE = "logistic_regression_model.pkl"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading Adult Income dataset...")

df = pd.read_csv(
    DATASET_FILE,
    na_values=["?", " ?"]
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

# Remove accidental spaces from column names.

df.columns = df.columns.str.strip()


# ============================================================
# 4. CLEAN TARGET COLUMN
# ============================================================

TARGET = "income"

if TARGET not in df.columns:
    raise ValueError(
        "ERROR: 'income' column was not found in adult.csv"
    )


# Remove rows where income is missing

df = df.dropna(subset=[TARGET])


# Clean income values

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.replace(".", "", regex=False)
)


# ============================================================
# 5. CONVERT TARGET TO BINARY VALUES
# ============================================================

# Standard Adult dataset:
#
# <=50K  → 0
# >50K   → 1

df[TARGET] = df[TARGET].map({
    "<=50K": 0,
    ">50K": 1
})


# Remove rows that could not be converted

df = df.dropna(subset=[TARGET])

df[TARGET] = df[TARGET].astype(int)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]


print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 9. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 10. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 11. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ============================================================
# 12. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Probability of belonging to >50K class

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 15. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "<=50K",
            ">50K"
        ]
    )
)


# ============================================================
# 18. SHOW SAMPLE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability >50K": y_probability
})


print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    prediction_results
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 19. RETRAIN USING COMPLETE DATASET
# ============================================================

print("\nRetraining final model using complete dataset...")

model.fit(
    X,
    y
)

print("Final model trained successfully!")


# ============================================================
# 20. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved successfully:")
print(MODEL_FILE)


# ============================================================
# 21. EXAMPLE PREDICTION
# ============================================================

example = X.iloc[[0]]

example_prediction = model.predict(
    example
)[0]

example_probability = model.predict_proba(
    example
)[0][1]


print("\n")
print("=" * 60)
print("EXAMPLE PREDICTION")
print("=" * 60)

if example_prediction == 1:
    print("Predicted Income: >50K")
else:
    print("Predicted Income: <=50K")

print(
    f"Probability of >50K: "
    f"{example_probability * 100:.2f}%"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("Model       : Logistic Regression")
print("Dataset     : Adult Income")
print("Target      : income")
print("Task        : Binary Classification")
print("Class 0     : <=50K")
print("Class 1     : >50K")
print(f"Accuracy    : {accuracy * 100:.2f}%")
print("Model file  :", MODEL_FILE)

