import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart.csv")
print("First 5 rows:")
print(df.head())

# STEP 4: Encode categorical features (if any)
label_encoder = LabelEncoder()
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = label_encoder.fit_transform(df[col])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# STEP 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 7: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# STEP 8: Predict
y_pred = model.predict(X_test)

# STEP 9: Evaluate
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n  Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()