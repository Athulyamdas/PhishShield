import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json

df = pd.read_csv("preprocessed_data.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# train test split
X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Ridge and Lasso

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_test, lasso_pred)

print("Ridge Regression Performance:")
print(f"MSE: {ridge_mse}, \n RMSE: {ridge_rmse},\n R² Score: {ridge_r2}")

print("\nLasso Regression Performance:")
print(f"MSE: {lasso_mse},\n RMSE: {lasso_rmse},\n R² Score: {lasso_r2}")

#Convert Predictions to Binary Class Labels
ridge_pred_binary = [1 if pred >= 0.5 else 0 for pred in ridge_pred]
lasso_pred_binary = [1 if pred >= 0.5 else 0 for pred in lasso_pred]

ridge_accuracy = accuracy_score(y_test, ridge_pred_binary)
ridge_precision = precision_score(y_test, ridge_pred_binary)
ridge_recall = recall_score(y_test, ridge_pred_binary)
ridge_f1 = f1_score(y_test, ridge_pred_binary)
ridge_conf_matrix = confusion_matrix(y_test, ridge_pred_binary)

lasso_accuracy = accuracy_score(y_test, lasso_pred_binary)
lasso_precision = precision_score(y_test, lasso_pred_binary)
lasso_recall = recall_score(y_test, lasso_pred_binary)
lasso_f1 = f1_score(y_test, lasso_pred_binary)
lasso_conf_matrix = confusion_matrix(y_test, lasso_pred_binary)

print("Ridge Regression - Classification Metrics:")
print(f"Accuracy: {ridge_accuracy:.4f},\n Precision: {ridge_precision:.4f},\n Recall: {ridge_recall:.4f},\n F1-score: {ridge_f1:.4f}\n")
print("Confusion Matrix:\n", ridge_conf_matrix)

print("\nLasso Regression - Classification Metrics:")
print(f"Accuracy: {lasso_accuracy:.4f},\n Precision: {lasso_precision:.4f},\n Recall: {lasso_recall:.4f},\n F1-score: {lasso_f1:.4f}\n")
print("Confusion Matrix:\n", lasso_conf_matrix)

print("\nClassification Report:Ridge")
print(classification_report(y_test, ridge_pred_binary))

print("\nClassification Report:Lasso")
print(classification_report(y_test, lasso_pred_binary))

# Feature Importance
ridge_feature_importance = np.abs(ridge_model.coef_)
ridge_feature_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': ridge_feature_importance})
ridge_feature_df_top_20 = ridge_feature_df.sort_values(by='Importance', ascending=False).head(20)
print("Top 20 Features (Ridge):")
print(ridge_feature_df_top_20)

lasso_feature_importance = np.abs(lasso_model.coef_)
print(type(lasso_feature_importance))
lasso_feature_importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": lasso_feature_importance.flatten() })
lasso_feature_df_top_20 = lasso_feature_importance_df.sort_values(by='Importance', ascending=False).head(20)
lasso_feature_df_top_20['Importance'] = lasso_feature_df_top_20['Importance'].apply(lambda x: round(x, 6))
print("Top 20 Features (Lasso):")
print(lasso_feature_df_top_20)

# Plot Ridge Top 20 Features
plt.figure(figsize=(12, 6))
sns.barplot(x=ridge_feature_df_top_20['Importance'], y=ridge_feature_df_top_20['Feature'], palette='coolwarm')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 20 Features - Ridge Regression")
plt.show()

# Plot Lasso Top 20 Features
plt.figure(figsize=(12, 6))
sns.barplot(x=lasso_feature_df_top_20['Importance'], y=lasso_feature_df_top_20['Feature'], palette='coolwarm')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 20 Features - Lasso Regression")
plt.show()

#storing the data to json file
try:
    with open("model_metrics.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    model_results = {}

model_results["Ridge Regression"] = [
    ridge_accuracy,ridge_precision,ridge_recall,ridge_f1
]

model_results["Lasso Regression"] = [
    lasso_accuracy,lasso_precision,lasso_recall,lasso_f1
]

with open("model_metrics.json", "w") as f:
    json.dump(model_results, f, indent=4)