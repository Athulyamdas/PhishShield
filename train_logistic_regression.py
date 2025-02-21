import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json

df = pd.read_csv("preprocessed_data.csv")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


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

#Logistic Regression

log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)
f1_log_reg = f1_score(y_test, y_pred_log_reg)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)

print(f"Accuracy: {accuracy_log_reg}")
print(f"Precision: {precision_log_reg}")
print(f"Recall: {recall_log_reg}")
print(f"F1-Score: {f1_log_reg}")
print("\nConfusion Matrix:")
print(conf_matrix_log_reg)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log_reg))

#feature importance score
feature_importance_log_reg = np.abs(log_reg.coef_[0])
feature_importance_df_log_reg = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance_log_reg})
feature_importance_df_log_reg = feature_importance_df_log_reg.sort_values(by='Importance', ascending=False)
top_20_features_log_reg = feature_importance_df_log_reg.head(20)
print(top_20_features_log_reg)

# Visualizing Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=top_20_features_log_reg['Importance'], y=top_20_features_log_reg['Feature'], palette='coolwarm')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 20 Important Features - Logistic Regression")
plt.show()

#storing the data to json file
try:
    with open("model_metrics.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    model_results = {}

model_results["Logistic Regression"] = [
    accuracy_log_reg, precision_log_reg, recall_log_reg, f1_log_reg
]

with open("model_metrics.json", "w") as f:
    json.dump(model_results, f, indent=4)


with open("model_metrics.json", "r") as f:
    model_results = json.load(f)


