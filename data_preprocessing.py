import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import StandardScaler

phishing_ds = pd.read_csv("phishing_url.csv")
df = pd.read_csv("phishing_url.csv")
print(df)

print(df.describe())

print(df.describe().transpose())

print(df.info())

print(df.isnull().sum())

# Removing FILENAME, URL, and Domain as they do not contribute to predictive modeling.
df = df.drop(columns=["FILENAME", "URL", "Domain"])
print(df.info())

# Label Encoding of the columns Title and TLD

tld_encoder = LabelEncoder()
title_encoder = LabelEncoder()
df["TLD"] = tld_encoder.fit_transform(df["TLD"])
df["Title"] = title_encoder.fit_transform(df["Title"])
print(df["TLD"])
print(df["Title"])

#saving Encoders using pickcle

with open("tld_encoder.pkl", "wb") as f:
    pickle.dump(tld_encoder, f)

with open("title_encoder.pkl", "wb") as f:
    pickle.dump(title_encoder, f)

print(df.info())


#normalization of the features - URLSimilarityIndex, LetterRatioInURL, ObfuscationRatio

scaler = StandardScaler()
features_to_scale = [
    "URLLength", "DomainLength", "NoOfSubDomain", "LargestLineLength",
    "LetterRatioInURL", "NoOfLettersInURL", "NoOfDegitsInURL", "NoOfOtherSpecialCharsInURL",
    "URLSimilarityIndex", "ObfuscationRatio", "NoOfObfuscatedChar",
    "LineOfCode", "DomainTitleMatchScore"
]
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print(df)

#saving preprocessed data to csv

df.to_csv("preprocessed_data.csv",index=False)

