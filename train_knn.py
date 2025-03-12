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
from sklearn.neighbors import KNeighborsClassifier


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

#KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

print(f"Accuracy: {accuracy_knn:.4f}")
print(f"Precision: {precision_knn:.4f}")
print(f"Recall: {recall_knn:.4f}")
print(f"F1-score: {f1_knn:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix_knn)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# Using Distance-Based Feature Importance
distances, indices = knn_model.kneighbors(X_test)
feature_importance_knn = np.zeros(X_train.shape[1])

for i, idx in enumerate(indices):
    neighbor_samples = X_train.iloc[idx] #Extracts 5 nearest training neighbors
    feature_diffs = np.abs(X_test.iloc[i] - neighbor_samples).mean(axis=0)
    feature_importance_knn += feature_diffs.values
print(feature_importance_knn)

feature_importance_knn /= len(X_test) #Computes the average importance score across all test samples (normalizing)

knn_feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance_knn
})
knn_feature_top_20 = knn_feature_importance_df.sort_values(by='Importance', ascending=False).head(20)
print("Top 20 Features (KNN - Distance-Based Importance):")
print(knn_feature_top_20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=knn_feature_top_20, palette="coolwarm")
plt.title("Top 20 Important Features - KNN")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()

#storing the data to json file
try:
    with open("model_metrics.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    model_results = {}

model_results["KNN"] = [
    accuracy_knn, precision_knn, recall_knn, f1_knn
]

with open("model_metrics.json", "w") as f:
    json.dump(model_results, f, indent=4)
