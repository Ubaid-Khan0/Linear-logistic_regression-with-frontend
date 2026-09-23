
# ============================================================
# BUILDING ENERGY - LINEAR REGRESSION
# Predicting Continuous Site EUI
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. FILE PATHS
# ============================================================

TRAIN_FILE = "building-energy/train.csv"
TEST_FILE = "building-energy/test.csv"
SAMPLE_FILE = "building-energy/sample_solution.csv"

MODEL_FILE = "linear_regression_model.pkl"
PREDICTION_FILE = "predictions.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
sample_df = pd.read_csv(SAMPLE_FILE)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample solution shape:", sample_df.shape)


# ============================================================
# 3. DISPLAY DATA INFORMATION
# ============================================================

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTest columns:")
print(test_df.columns.tolist())


# ============================================================
# 4. SEPARATE TARGET AND FEATURES
# ============================================================

# site_eui is the continuous value we want to predict.

TARGET = "site_eui"
ID_COLUMN = "id"


# Make sure the target exists in training data
if TARGET not in train_df.columns:
    raise ValueError(
        "ERROR: 'site_eui' was not found in train.csv"
    )


# Separate target
y = train_df[TARGET]


# Remove target and ID from training features
X = train_df.drop(
    columns=[TARGET, ID_COLUMN],
    errors="ignore"
)


# Remove ID from test features
X_test_final = test_df.drop(
    columns=[ID_COLUMN],
    errors="ignore"
)


print("\nTraining features:", X.shape)
print("Target:", y.shape)
print("Test features:", X_test_final.shape)


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumber of numerical features:",
      len(numerical_features))

print("Number of categorical features:",
      len(categorical_features))

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 6. NUMERICAL PREPROCESSING
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
# 7. CATEGORICAL PREPROCESSING
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
# 8. COMBINE PREPROCESSING
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
# 9. CREATE LINEAR REGRESSION PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# 10. SPLIT TRAINING DATA
# ============================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining rows:", X_train.shape[0])
print("Validation rows:", X_validation.shape[0])


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ============================================================
# 12. VALIDATION PREDICTIONS
# ============================================================

validation_predictions = model.predict(
    X_validation
)


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)

r2 = r2_score(
    y_validation,
    validation_predictions
)


print("\n")
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 14. SHOW SAMPLE VALIDATION PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Site EUI": y_validation.values,
    "Predicted Site EUI": validation_predictions
})

print("\n")
print("=" * 50)
print("SAMPLE VALIDATION PREDICTIONS")
print("=" * 50)

print(
    results.head(10).to_string(index=False)
)


# ============================================================
# 15. RETRAIN MODEL USING ALL TRAINING DATA
# ============================================================

print("\n")
print("Retraining model using complete training dataset...")

model.fit(
    X,
    y
)

print("Final model trained!")


# ============================================================
# 16. PREDICT TEST DATA
# ============================================================

print("\nPredicting site_eui for test.csv...")

test_predictions = model.predict(
    X_test_final
)


# ============================================================
# 17. CREATE PREDICTION FILE
# ============================================================

# Keep the original IDs from test.csv
prediction_df = pd.DataFrame({
    "id": test_df[ID_COLUMN],
    "site_eui": test_predictions
})


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print("\nPrediction file created:")
print(PREDICTION_FILE)


# ============================================================
# 19. DISPLAY TEST PREDICTIONS
# ============================================================

print("\n")
print("=" * 50)
print("TEST PREDICTIONS")
print("=" * 50)

print(
    prediction_df.head(10).to_string(index=False)
)


# ============================================================
# 20. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\nTrained model saved as:")
print(MODEL_FILE)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 50)
print("PROJECT COMPLETED")
print("=" * 50)

print("Model          : Linear Regression")
print("Target         : site_eui")
print("Prediction     : Continuous numerical value")
print("Training file  : train.csv")
print("Testing file   : test.csv")
print("Output file    : predictions.csv")
print("Model file     : linear_regression_model.pkl")

print("\nExample predicted values:")

for value in test_predictions[:5]:
    print(f"{value:.2f}")

