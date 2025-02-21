import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("preprocessed_data.csv")

with open("top_20_features.json", "r") as f:
    top_20_features = json.load(f)

X_all = df.drop(columns=["label"])
X_top20 = df[top_20_features]
y = df["label"]

X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
X_train_top20, X_test_top20, y_train_top20, y_test_top20 = train_test_split(X_top20, y, test_size=0.2, random_state=42)

rf_all = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_all.fit(X_train_all, y_train)
y_pred_all = rf_all.predict(X_test_all)

rf_top20 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_top20.fit(X_train_top20, y_train)
y_pred_top20 = rf_top20.predict(X_test_top20)

def evaluate_model(y_true, y_pred):
    return{
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

metrics_all = evaluate_model(y_test, y_pred_all)
metrics_top20 = evaluate_model(y_test, y_pred_top20)

comparison_results = {
    "Random Forest (All Features)": metrics_all,
    "Random Forest (Top 20 Features)": metrics_top20
}

with open("random_forest_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=4)


print("Performance Comparison:")
print(pd.DataFrame(comparison_results))
