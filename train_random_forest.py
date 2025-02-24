import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json
import pickle

df = pd.read_csv("preprocessed_data.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

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

# Random Forest

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf= rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest Precision: {precision_rf:.4f}")
print(f"Random Forest Recall: {recall_rf:.4f}")
print(f"Random Forest F1 Score: {f1_rf:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_rf}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()


# Feature Importance
rf_feature_importance = rf_model.feature_importances_
rf_feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_feature_importance
})
rf_feature_top_20 = rf_feature_importance_df.sort_values(by='Importance', ascending=False).head(20)
print("Top 20 Features (Random Forest):")
print(rf_feature_top_20)
#print(type(rf_feature_top_20))

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_feature_top_20, palette="coolwarm")
plt.title("Top 20 Important Features - Random Forest")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()

#storing the data to json file
try:
    with open("model_metrics.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    model_results = {}

model_results["Random Forest"] = [
    accuracy_rf, precision_rf, recall_rf, f1_rf
]

with open("model_metrics.json", "w") as f:
    json.dump(model_results, f, indent=4)

#saving top 20 features into a json file
top_20_features_list = rf_feature_top_20.iloc[:, 0].tolist()

with open("top_20_features.json", "w") as f:
    json.dump(top_20_features_list, f, indent=4)

print("Top 20 features saved to 'top_20_features.json'")

# save trained Random Forest model as a .pkl file
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Random Forest model saved successfully!")