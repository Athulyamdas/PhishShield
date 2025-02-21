import json
import pandas as pd

with open("model_metrics.json", "r") as f:
    model_results = json.load(f)

df_results = pd.DataFrame.from_dict(model_results, orient="index", columns=["Accuracy", "Precision", "Recall", "F1-Score"])
df_results = df_results.sort_values(by="F1-Score", ascending=False)

print("Sorted Model Performance (by F1-Score):\n")
print(df_results)