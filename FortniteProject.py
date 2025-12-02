import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, auc
)

# -------------------------
# Load & clean data
# -------------------------
main_df = pd.read_csv("Fortnite Statistics.csv")

# Remove unused columns
revised_df = main_df.drop(columns=['Date', 'Time of Day'])

# Convert Accuracy from "xx%" string → float (0–1)
revised_df['Accuracy'] = (
    revised_df['Accuracy'].str.replace('%', '').astype(float) / 100
)

# -------------------------
# LINEAR REGRESSION
# Predict exact placement
# -------------------------
X_reg = revised_df[['Eliminations', 'Damage to Players', 'Accuracy']]
y_reg = revised_df['Placed']

# Train/test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train linear regression
reg_model = LinearRegression()
reg_model.fit(X_train_reg_scaled, y_train_reg)

# Predict
y_pred_reg = reg_model.predict(X_test_reg_scaled)

# Evaluate regression
print("----- Linear Regression Metrics -----")
print("R²:", r2_score(y_test_reg, y_pred_reg))
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))

# -------------------------
# BINARY CLASSIFICATION
# Top 10 (1) vs Not Top 10 (0)
# -------------------------
revised_df['Top10'] = (revised_df['Placed'] <= 10).astype(int)

X_clf = revised_df[['Eliminations', 'Damage to Players', 'Accuracy']]
y_clf = revised_df['Top10']

# Stratified split to maintain class balance
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Scale features
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Logistic regression (binary)
clf_model = LogisticRegression(max_iter=500)
clf_model.fit(X_train_clf_scaled, y_train_clf)

# Predict
y_pred_clf = clf_model.predict(X_test_clf_scaled)

# Evaluate classification
print("\n----- Binary Classification Metrics (Top 10 vs Not Top 10) -----")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Precision:", precision_score(y_test_clf, y_pred_clf, zero_division=0))
print("Recall:", recall_score(y_test_clf, y_pred_clf, zero_division=0))
print("F1 Score:", f1_score(y_test_clf, y_pred_clf, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=['Not Top 10','Top 10'], zero_division=0))

# Quick check for unique values
print("Unique classes in test target:", np.unique(y_test_clf))
print("Unique classes in predictions:", np.unique(y_pred_clf))

#visualize model error for linear
plt.figure(figsize=(6,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel("Actual Placement")
plt.ylabel("Predicted Placement")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

#ROC Curve for logistic
y_prob = clf_model.predict_proba(X_test_clf_scaled)[:,1]  # probability for class 1
fpr, tpr, thresholds = roc_curve(y_test_clf, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend()
plt.show()
