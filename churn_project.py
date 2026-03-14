import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# 1. Load dataset
# -----------------------------
file_path = r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found at: {file_path}\n"
        "Make sure the CSV file exists in the correct folder."
    )

df = pd.read_csv(file_path)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Shape ===")
print(df.shape)

print("\n=== Column Info ===")
print(df.info())

print("\n=== Missing Values Before Cleaning ===")
print(df.isnull().sum())

# -----------------------------
# 2. Data cleaning
# -----------------------------
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill object missing values with mode
object_cols = df.select_dtypes(include=["object"]).columns
for col in object_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())
print("\nTotal missing after cleaning:", df.isnull().sum().sum())

# -----------------------------
# 3. Create output folder
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 4. EDA on original data
# -----------------------------

# Churn distribution
plt.figure(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/churn_distribution.png")
plt.close()

# Monthly charges by churn
if "MonthlyCharges" in df.columns:
    plt.figure(figsize=(7, 5))
    df.boxplot(column="MonthlyCharges", by="Churn")
    plt.title("Monthly Charges by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig("outputs/monthlycharges_by_churn.png")
    plt.close()

# Tenure by churn
if "tenure" in df.columns:
    plt.figure(figsize=(7, 5))
    df.boxplot(column="tenure", by="Churn")
    plt.title("Tenure by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Tenure")
    plt.tight_layout()
    plt.savefig("outputs/tenure_by_churn.png")
    plt.close()

# Churn by contract type
if "Contract" in df.columns:
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    contract_churn.plot(kind="bar", figsize=(8, 5))
    plt.title("Churn Percentage by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage")
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig("outputs/churn_by_contract.png")
    plt.close()

# -----------------------------
# 5. Encode categorical columns
# -----------------------------
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

print("\n=== Encoded Dataset Preview ===")
print(df_encoded.head())

# -----------------------------
# 6. Correlation heatmap
# -----------------------------
plt.figure(figsize=(14, 10))
sns.heatmap(df_encoded.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# -----------------------------
# 7. Split features and target
# -----------------------------
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== Train Test Split ===")
print("Training set shape:", X_train.shape)
print("Testing set shape :", X_test.shape)

# -----------------------------
# 8. Scale data for Logistic Regression
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 9. Train Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -----------------------------
# 10. Train Logistic Regression
# -----------------------------
lr_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

# -----------------------------
# 11. Model comparison
# -----------------------------
print("\n=== Model Comparison ===")
print(f"Random Forest Accuracy     : {rf_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

comparison_df = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "Accuracy": [rf_acc, lr_acc]
})

plt.figure(figsize=(7, 5))
plt.bar(comparison_df["Model"], comparison_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# -----------------------------
# 12. Classification reports
# -----------------------------
print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, rf_pred))

print("\n=== Logistic Regression Classification Report ===")
print(classification_report(y_test, lr_pred))

# -----------------------------
# 13. Confusion Matrix - Random Forest
# -----------------------------
rf_cm = confusion_matrix(y_test, rf_pred)
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm)
rf_disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_random_forest.png")
plt.close()

# -----------------------------
# 14. Confusion Matrix - Logistic Regression
# -----------------------------
lr_cm = confusion_matrix(y_test, lr_pred)
lr_disp = ConfusionMatrixDisplay(confusion_matrix=lr_cm)
lr_disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_logistic_regression.png")
plt.close()

# -----------------------------
# 15. Random Forest feature importance
# -----------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\n=== Top 10 Important Features (Random Forest) ===")
print(importances.head(10))

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/top_10_feature_importance.png")
plt.close()

# -----------------------------
# 16. Save model results to CSV
# -----------------------------
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "RandomForest_Prediction": rf_pred,
    "LogisticRegression_Prediction": lr_pred
})
results_df.to_csv("outputs/prediction_results.csv", index=False)

# -----------------------------
# 17. Final message
# -----------------------------
print("\nProject completed successfully.")
print("Files saved inside the outputs folder:")
print("""
- churn_distribution.png
- monthlycharges_by_churn.png
- tenure_by_churn.png
- churn_by_contract.png
- correlation_heatmap.png
- model_comparison.png
- confusion_matrix_random_forest.png
- confusion_matrix_logistic_regression.png
- top_10_feature_importance.png
- prediction_results.csv
""")