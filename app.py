from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
from feature_extraction import extract_features_from_live_site

app = Flask(__name__)

# Load dataset
df = pd.read_csv("preprocessed_data.csv")
original_df = pd.read_csv("phishing_url.csv")
df = df.drop(columns=["label"])

# Load trained model
with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load label encoders
with open("tld_encoder.pkl", "rb") as f:
    tld_encoder = pickle.load(f)

with open("title_encoder.pkl", "rb") as f:
    title_encoder = pickle.load(f)


def safe_transform(encoder, values):
    """
    Transforms values using LabelEncoder.
    If a value is unseen, it assigns a placeholder value.
    """
    unseen_label = len(encoder.classes_)  # Assign a new label for unseen values
    encoder_classes = list(encoder.classes_)
    transformed = []

    for value in values:
        if value in encoder_classes:
            transformed.append(encoder.transform([value])[0])
        else:
            transformed.append(unseen_label)  # Placeholder for unseen values

    return np.array(transformed)


def get_feature_values(input_url):
    if input_url in original_df["URL"].values:
        index = original_df[original_df["URL"] == input_url].index[0]
        feature_values = original_df.drop(columns=["FILENAME", "URL", "Domain", "label"]).iloc[index]

        if "TLD" in feature_values.index:
            feature_values["TLD"] = tld_encoder.transform([feature_values["TLD"]])[0]

        if "Title" in feature_values.index:
            feature_values["Title"] = title_encoder.transform([feature_values["Title"]])[0]

        feature_values = feature_values.astype(float).to_numpy().reshape(1, -1)
        return feature_values

    else:
        # URL not found, so extract features dynamically
        extracted_features = extract_features_from_live_site(input_url)
        if extracted_features:
            print("Features extracted successfully from live site.")
            extracted_df = pd.DataFrame([extracted_features], columns=df.columns)

            # Handle encoding for TLD and Title
            if "TLD" in extracted_df.columns:
                extracted_df["TLD"] = tld_encoder.transform(extracted_df["TLD"])

            if "Title" in extracted_df.columns:
                extracted_df["Title"] = safe_transform(title_encoder, extracted_df["Title"])

            return extracted_df.to_numpy().reshape(1, -1)
        else:
            print("Failed to extract features from live site.")
            return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_url = request.form["url"]

    feature_values = get_feature_values(input_url)

    if feature_values is not None:
        print("Feature values found. Making prediction...")

        feature_values_df = pd.DataFrame(feature_values, columns=df.columns)

        prediction = rf_model.predict(feature_values_df)[0]
        print(f"Prediction result: {prediction}")

        result = "Legitimate. You are Safe !!!" if prediction == 1 else "Phishing!! Be Careful !!!"
        print(result)
    else:
        print("Feature values are None. Returning default message.")
        result = "Sorry!! Couldn't Parse this link"

    print("Returning prediction result to frontend.")

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)