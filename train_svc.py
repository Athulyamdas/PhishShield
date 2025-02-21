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
from sklearn.svm import SVC

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


#SVC
svc_model = SVC(kernel='linear', random_state=42)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
precision_svc = precision_score(y_test, y_pred_svc)
recall_svc = recall_score(y_test, y_pred_svc)
f1_svc = f1_score(y_test, y_pred_svc)
conf_matrix_svc = confusion_matrix(y_test, y_pred_svc)
print(f"SVM Accuracy: {accuracy_svc:.4f}")
print(f"SVM Precision: {precision_svc:.4f}")
print(f"SVM Recall: {recall_svc:.4f}")
print(f"SVM F1 Score: {f1_svc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_svc}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svc))

#visualization
svc_feature_importance = np.abs(svc_model.coef_).flatten()
svc_feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': svc_feature_importance
})
svc_feature_top_20 = svc_feature_importance_df.sort_values(by='Importance', ascending=False).head(20)
print("Top 20 Features (SVC):")
print(svc_feature_top_20)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=svc_feature_top_20, palette="coolwarm")
plt.title("Top 20 Important Features - SVC")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()

#storing the data to json file
try:
    with open("model_metrics.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    model_results = {}

model_results["SVC"] = [
    accuracy_svc, precision_svc, recall_svc, f1_svc
]

with open("model_metrics.json", "w") as f:
    json.dump(model_results, f, indent=4)
