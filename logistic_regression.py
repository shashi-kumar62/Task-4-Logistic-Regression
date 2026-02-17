# ==========================================================
# Task 4: Classification with Logistic Regression
# Breast Cancer Wisconsin Dataset
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

# ==========================================================
# 1. Load Dataset
# ==========================================================

df = pd.read_csv("data.csv")

print("===== FIRST 5 ROWS =====")
print(df.head())

# ==========================================================
# 2. Data Cleaning
# ==========================================================

# Drop unnecessary columns
df = df.drop(["id", "Unnamed: 32"], axis=1)

# Convert diagnosis to binary (M=1, B=0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("\nClass Distribution:")
print(df["diagnosis"].value_counts())

# ==========================================================
# 3. Split Features & Target
# ==========================================================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# 4. Standardize Features
# ==========================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# 5. Train Logistic Regression Model
# ==========================================================

model = LogisticRegression()
model.fit(X_train, y_train)

# ==========================================================
# 6. Predictions
# ==========================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================================
# 7. Evaluation Metrics
# ==========================================================

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\nPrecision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================================
# 8. ROC Curve
# ==========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# ==========================================================
# 9. Threshold Tuning
# ==========================================================

print("\n===== THRESHOLD TUNING =====")

for threshold in [0.3, 0.5, 0.7]:
    y_custom = (y_prob >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(y_test, y_custom))
    print("Recall:", recall_score(y_test, y_custom))

# ==========================================================
# 10. Sigmoid Function Plot
# ==========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.figure()
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.show()

print("\nTask Completed Successfully 🚀")

