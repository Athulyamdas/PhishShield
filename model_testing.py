import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("preprocessed_data.csv")

X = df.drop(columns=["label"])
y = df["label"]

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

y_pred = rf_model.predict(X)

sample_input = X.head(10)
sample_pred = rf_model.predict(sample_input)
print("Sample Inputs:")
print(sample_input)
print("\nPredictions for Sample Inputs:")
print(sample_pred)

train_accuracy = rf_model.score(X, y)
print(f"\nTrain Accuracy: {train_accuracy:.4f}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
test_accuracy = rf_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

