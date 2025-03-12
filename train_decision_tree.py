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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz


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


# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10,min_samples_split=20,min_samples_leaf=10,
                                  max_features='sqrt',
                                  ccp_alpha=0.01,random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-score: {f1_dt:.4f}")
print(f"Mean Squared Error (MSE): {mse_dt:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_dt:.4f}")
print(f"R² Score: {r2_dt:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix_dt)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

feature_importance = dt_model.feature_importances_
dt_feature_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
})
dt_feature_df_top_20 = dt_feature_df.sort_values(by='Importance', ascending=False).head(20)
print(dt_feature_df_top_20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=dt_feature_df_top_20, palette="coolwarm")
plt.title("Top 20 Important Features - Decision Tree")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()

#storing the data to json file
try:
    with open("model_metrics.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    model_results = {}

model_results["Decision Tree"] = [
    accuracy_dt, precision_dt, recall_dt, f1_dt
]

with open("model_metrics.json", "w") as f:
    json.dump(model_results, f, indent=4)

